// Copyright 2026 Bing Xie
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Pending-request manager: wraps `Tree` with an rid → tokens map so callers
//! can `remove(rid)` without re-supplying the token sequence, and exposes the
//! whole thing as a PyO3 class.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::tree::{NodeId, Rid, Token, Tree, ROOT};

pub struct PendingTree {
    tree: Tree,
    /// rid → token path. Kept so remove(rid) doesn't need tokens from the caller.
    pub(crate) paths: FxHashMap<Rid, Vec<Token>>,
}

impl PendingTree {
    pub fn new() -> Self {
        Self {
            tree: Tree::new(),
            paths: FxHashMap::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn contains(&self, rid: Rid) -> bool {
        self.paths.contains_key(&rid)
    }

    /// Insert a request. Returns true if newly inserted, false if rid was
    /// already present (no-op — caller should ensure rids are unique).
    pub fn insert(&mut self, rid: Rid, tokens: Vec<Token>) -> bool {
        if self.paths.contains_key(&rid) {
            return false;
        }
        self.tree.insert(rid, &tokens);
        self.paths.insert(rid, tokens);
        true
    }

    /// Remove a request by rid. Returns true if it was present.
    pub fn remove(&mut self, rid: Rid) -> bool {
        let Some(tokens) = self.paths.remove(&rid) else {
            return false;
        };
        let ok = self.tree.remove(rid, &tokens);
        debug_assert!(ok, "path/tree inconsistency for rid={}", rid);
        ok
    }

    /// Length of the longest prefix this rid's tokens share with at least one
    /// *other* pending request. Returns 0 if the rid is not in the tree or
    /// has no shared prefix.
    pub fn longest_shared_prefix(&self, rid: Rid) -> usize {
        let Some(tokens) = self.paths.get(&rid) else {
            return 0;
        };
        // Threshold 2: pending_count counts `rid` itself + at least one other.
        self.tree.deepest_with_count(tokens, 2)
    }

    /// Length of the longest prefix `tokens` shares with any request currently
    /// in the tree. Useful for pre-insert queries.
    pub fn match_prefix(&self, tokens: &[Token]) -> usize {
        self.tree.deepest_with_count(tokens, 1)
    }

    pub fn node_count(&self) -> usize {
        self.tree.len()
    }

    /// Cluster info for `rid`. See `Tree::cluster_info` for semantics.
    pub fn cluster_info(&self, rid: Rid) -> Option<(NodeId, usize, u32)> {
        let tokens = self.paths.get(&rid)?;
        self.tree.cluster_info(rid, tokens)
    }
}

/// Python-facing class. Rids are exposed as `int` (u64). Tokens are `list[int]`
/// (u32). Methods validate inputs and raise clean Python exceptions.
#[pyclass(name = "PendingTree", module = "peek._core")]
pub struct PyPendingTree {
    inner: PendingTree,
}

#[pymethods]
impl PyPendingTree {
    #[new]
    fn new() -> Self {
        Self {
            inner: PendingTree::new(),
        }
    }

    /// Insert a pending request. Raises ValueError if rid is already present.
    fn insert(&mut self, rid: u64, tokens: Vec<u32>) -> PyResult<()> {
        if !self.inner.insert(rid, tokens) {
            return Err(PyValueError::new_err(format!(
                "rid {} already present",
                rid
            )));
        }
        Ok(())
    }

    /// Remove a pending request by rid. Raises KeyError if not present.
    fn remove(&mut self, rid: u64) -> PyResult<()> {
        if !self.inner.remove(rid) {
            return Err(PyKeyError::new_err(rid));
        }
        Ok(())
    }

    /// Same as `remove` but returns False instead of raising if rid is absent.
    fn discard(&mut self, rid: u64) -> bool {
        self.inner.remove(rid)
    }

    /// Bulk insert. Equivalent to calling `insert` once per `(rid, tokens)`
    /// tuple, but amortizes the Python↔Rust crossing over the whole batch.
    /// All-or-nothing: if any rid is already present, raises ValueError and
    /// makes no changes.
    fn insert_many(&mut self, items: Vec<(u64, Vec<u32>)>) -> PyResult<()> {
        // Pre-flight: reject the whole batch if any rid is already present.
        // Cheap dup-check within the batch too (same rid twice in one call is
        // a caller bug).
        for (rid, _) in &items {
            if self.inner.contains(*rid) {
                return Err(PyValueError::new_err(format!(
                    "rid {} already present",
                    rid
                )));
            }
        }
        let mut seen: rustc_hash::FxHashSet<u64> =
            rustc_hash::FxHashSet::default();
        seen.reserve(items.len());
        for (rid, _) in &items {
            if !seen.insert(*rid) {
                return Err(PyValueError::new_err(format!(
                    "rid {} appears twice in batch",
                    rid
                )));
            }
        }
        // Reserve path map capacity to avoid per-insert resizes.
        self.inner.paths.reserve(items.len());
        for (rid, tokens) in items {
            let ok = self.inner.insert(rid, tokens);
            debug_assert!(ok);
        }
        Ok(())
    }

    /// Bulk remove. Returns the number of rids that were present and removed.
    /// Unlike `remove`, missing rids are silently skipped — matches `discard`
    /// semantics for batch convenience.
    fn remove_many(&mut self, rids: Vec<u64>) -> usize {
        let mut n = 0;
        for rid in rids {
            if self.inner.remove(rid) {
                n += 1;
            }
        }
        n
    }

    fn __contains__(&self, rid: u64) -> bool {
        self.inner.contains(rid)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Longest prefix rid's tokens share with at least one *other* pending
    /// request. Returns 0 if rid is not tracked.
    fn longest_shared_prefix(&self, rid: u64) -> usize {
        self.inner.longest_shared_prefix(rid)
    }

    /// Longest prefix `tokens` shares with any request currently in the tree.
    fn match_prefix(&self, tokens: Vec<u32>) -> usize {
        self.inner.match_prefix(&tokens)
    }

    /// Internal node count (for debugging / tests).
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Fast O(#root-children) check: does any pending rid share even its
    /// first token with another pending rid? When False, peek's dualwalk /
    /// cluster_info cannot produce any scheduling signal beyond what stock
    /// LPM would — callers should bypass peek in that case.
    fn has_sharing(&self) -> bool {
        self.inner.tree.has_sharing()
    }

    /// Number of pending rids that share their first edge with at least one
    /// other pending rid. `shared + singleton == len(tree)`.
    fn shared_rid_count(&self) -> u32 {
        self.inner.tree.shared_rid_count()
    }

    /// Number of pending rids whose first edge is unique (nobody else shares
    /// the first token). Complement of `shared_rid_count`.
    fn singleton_rid_count(&self) -> u32 {
        self.inner.tree.singleton_rid_count()
    }

    /// Cluster-LPM per-request score:
    ///   score(rid) = Σ over ancestor chain of (pending_count × edge_length)
    /// where the chain runs from rid's terminator node up to root.
    ///
    /// Captures total pending-token-edges along rid's path — a dense,
    /// heavily-shared subtree yields high score; a lone singleton yields
    /// near zero. Single O(tree_size) DFS, O(1) per terminator annotation.
    fn compute_req_scores(&self) -> HashMap<u64, i64> {
        let mut result: HashMap<u64, i64> =
            HashMap::with_capacity(self.inner.paths.len());
        let mut stack: Vec<(NodeId, i64)> = Vec::new();
        stack.push((ROOT, 0));
        while let Some((node_id, score_here)) = stack.pop() {
            let node = self.inner.tree.node(node_id);
            for &rid in &node.terminators {
                result.insert(rid, score_here);
            }
            for &child_id in node.children.values() {
                let child = self.inner.tree.node(child_id);
                let contribution =
                    (child.pending_count as i64) * (child.edge.len() as i64);
                stack.push((child_id, score_here + contribution));
            }
        }
        result
    }

    /// Cluster info for `rid`: `(cluster_node_id, cluster_depth, cluster_size)`.
    /// Returns None if `rid` is absent or singleton. `cluster_node_id` is an
    /// opaque int stable within a mutation-free sequence — rids whose returned
    /// `cluster_node_id` match are in the same cluster.
    fn cluster_info(&self, rid: u64) -> Option<(u32, usize, u32)> {
        self.inner.cluster_info(rid)
    }

    /// Bulk cluster info for every pending rid. Returns a dict keyed by rid
    /// whose value is either `(cluster_node_id, cluster_depth, cluster_size)`
    /// or `None` for singletons. Collapses N Python↔Rust crossings into one.
    fn all_cluster_info(&self) -> HashMap<u64, Option<(u32, usize, u32)>> {
        let mut out: HashMap<u64, Option<(u32, usize, u32)>> = HashMap::default();
        for (&rid, tokens) in &self.inner.paths {
            out.insert(rid, self.inner.tree.cluster_info(rid, tokens));
        }
        out
    }

    /// Return the token sequence currently stored for `rid`, or None if absent.
    fn tokens(&self, rid: u64) -> Option<Vec<u32>> {
        self.inner.paths.get(&rid).cloned()
    }

    /// Number of pending rids whose token sequence begins with `path`.
    ///
    /// For demand-aware cache eviction: answers "how many future prefills
    /// would we force to recompute if we evict this path?" Zero means it's
    /// safe to evict; higher means more harmful.
    fn pending_demand(&self, path: Vec<u32>) -> u32 {
        self.inner.tree.pending_demand(&path)
    }

    /// Number of pending rids whose tokens END EXACTLY at `path`.
    ///
    /// Path-specific demand: unlike `pending_demand`, a shared ancestor with
    /// N fan-out sessions gets count 0 here (none terminate at the ancestor).
    /// Used for demand-aware eviction of sglang cache *leaves* so shared
    /// prefixes aren't amplified N× in the protection signal.
    fn terminators_at(&self, path: Vec<u32>) -> u32 {
        self.inner.tree.terminators_at(&path)
    }

    /// Walk `tokens` into the tree as deep as possible. Returns
    /// `(match_depth, Some(rid))` where `rid` is any pending rid whose path
    /// shares a `match_depth`-token prefix with `tokens`, or `(0, None)` if
    /// no pending rid matches even the first token.
    ///
    /// Used by peek's handoff mechanism to pair a near-finishing running req
    /// (given its full token prefix) with the best-overlap pending rid.
    fn longest_match_along(&self, tokens: Vec<u32>) -> (u32, Option<u64>) {
        self.inner.tree.longest_match_along(&tokens)
    }

    /// Record an observed completion: the req with tokens `tokens` emitted
    /// `output_len` decode tokens. Updates per-cluster EWMA at the deepest
    /// node matching `tokens`. The node stays alive after GC even if its
    /// pending count drops to zero, so `predict_decode` can query it later.
    fn record_decode(&mut self, tokens: Vec<u32>, output_len: u32) {
        self.inner.tree.record_decode(&tokens, output_len);
    }

    /// Predict decode length for a new req with token sequence `tokens`.
    /// Returns `(ewma, sample_count)` from the deepest tree node on `tokens`'s
    /// path whose sample count meets `min_samples`, or `None` if no qualifying
    /// ancestor exists.
    #[pyo3(signature = (tokens, min_samples = 3))]
    fn predict_decode(
        &self,
        tokens: Vec<u32>,
        min_samples: u32,
    ) -> Option<(f32, u32)> {
        self.inner.tree.predict_decode(&tokens, min_samples)
    }

    /// Dump every node of the tree as a flat list for out-of-Rust walking.
    /// Each entry: (node_id, parent_id, edge_tokens, terminators, pending_count).
    /// Root has node_id=0 and parent_id=0 (self-loop). Skips free slots.
    /// Used by the Python-side dual-walker to co-traverse sglang's KV cache.
    fn snapshot_for_walk(
        &self,
    ) -> Vec<(u32, u32, Vec<u32>, Vec<u64>, u32)> {
        let mut out = Vec::new();
        let free_set: std::collections::HashSet<u32> =
            self.inner.tree.free_slots().iter().copied().collect();
        for (idx, node) in self.inner.tree.nodes_slice().iter().enumerate() {
            let id = idx as u32;
            if id != ROOT && free_set.contains(&id) {
                continue;
            }
            let terminators: Vec<u64> = node.terminators.iter().copied().collect();
            out.push((id, node.parent, node.edge.clone(), terminators, node.pending_count));
        }
        out
    }

    /// Compute main-cache match length for every pending rid in a single
    /// dual-walk. `cache_match_fn` is called ONCE PER EDGE in peek's tree
    /// (not per rid). When the cache diverges within an edge, every rid in
    /// that subtree shares the same main_hit — no further cache queries for
    /// that branch. Returns `{rid: main_hit}` for every pending rid.
    ///
    /// `cache_match_fn(tokens: list[int]) -> int` must return the length of
    /// the longest prefix of `tokens` present in the external cache.
    ///
    /// `min_pending_count` (default 1) controls whether to descend into
    /// subtrees of low-sharing reqs. Set to 2 to skip singleton subtrees —
    /// where a req's tail diverges from all other pending reqs — on the
    /// assumption that exact per-req tails are rarely in the KV cache. This
    /// collapses typical per-pass cache calls from O(N) to O(#clusters) at
    /// the cost of missing tail-cache hits (uncommon in LLM traffic).
    #[pyo3(signature = (cache_match_fn, min_pending_count = 1))]
    fn compute_main_hits(
        &self,
        py: Python<'_>,
        cache_match_fn: Bound<'_, PyAny>,
        min_pending_count: u32,
    ) -> PyResult<HashMap<u64, usize>> {
        // Pre-size for the total pending rid count — every terminator gets
        // exactly one entry in the returned dict.
        let mut result: HashMap<u64, usize> =
            HashMap::with_capacity(self.inner.paths.len());
        let mut path: Vec<Token> = Vec::new();
        let mut stack: Vec<NodeId> = Vec::new();
        self.inner.dualwalk(
            ROOT,
            0,
            false,
            &mut path,
            &mut result,
            &mut stack,
            py,
            &cache_match_fn,
            min_pending_count,
        )?;
        Ok(result)
    }
}

/// Cluster-LPM sort: lexicographic 6-tuple stable sort.
///
/// Sort key per queue position:
///   (arrival_bucket, section_id, -main_hit, -req_score, -cluster_size, arrival_ns)
///
/// Semantics:
///   * arrival_bucket — primary (FCFS across windows; no starvation beyond W)
///   * section_id     — 0=warm, 1=cold pioneer, 2=cold sibling (deprio tail)
///   * -main_hit      — LPM primary within section
///   * -req_score     — cumulative (pending_count × edge_length) along ancestors
///   * -cluster_size  — shallow subtree with many reqs wins
///   * arrival_ns     — FCFS final tiebreak
///
/// Returns indices in admission order. Stable: tuple equality preserves input order.
#[pyfunction]
pub fn peek_clpm_sort_order(
    keys: Vec<(i64, i64, i64, i64, i64, i64)>,
) -> Vec<usize> {
    let n = keys.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
    idx
}

/// LPM-faithful stable sort of waiting-queue positions.
///
/// `keys[i]` is the (main_hit, is_deprioritized) pair for the i-th waiting
/// req. Returns indices in admission order:
///   - non-deprioritized reqs first, sorted by descending main_hit
///   - deprioritized reqs at the tail, in original arrival order
/// Ties preserve original queue position (stable sort).
///
/// Byte-identical to `sorted(queue, key=lambda r: float('inf') if depr else
/// -main_hit)` — note that Python treats ALL deprioritized reqs as having
/// the same key (+inf), so stable sort keeps them in arrival order. Within
/// non-deprio, main_hit breaks order; ties preserve arrival order.
#[pyfunction]
pub fn lpm_sort_order(keys: Vec<(i64, bool)>) -> Vec<usize> {
    let n = keys.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        let (mh_a, dp_a) = keys[a];
        let (mh_b, dp_b) = keys[b];
        match (dp_a, dp_b) {
            // Non-deprio before deprio.
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            // Both deprio: treat as equal (+inf key in Python) → stable
            // sort preserves arrival order. Crucial for LPM parity.
            (true, true) => Ordering::Equal,
            // Both non-deprio: larger main_hit first.
            (false, false) => (-mh_a).cmp(&(-mh_b)),
        }
    });
    idx
}

impl PendingTree {
    /// Recursive dual-walk. Skipping a child edge is only safe when we've
    /// already established a main_hit on the current path (i.e., queried at
    /// least once). For solitary reqs whose whole path is pc=1, the FIRST
    /// edge must still be queried to get the actual match. After that, pc=1
    /// tail subtrees can inherit the parent's main_hit without further
    /// queries (tails are typically cache-cold in LLM traffic).
    fn dualwalk(
        &self,
        node_id: NodeId,
        main_hit_at_node: usize,
        established: bool,
        path: &mut Vec<Token>,
        result: &mut HashMap<Rid, usize>,
        scratch: &mut Vec<NodeId>,
        py: Python<'_>,
        cache_match_fn: &Bound<'_, PyAny>,
        min_pending_count: u32,
    ) -> PyResult<()> {
        for &rid in &self.tree.node(node_id).terminators {
            result.insert(rid, main_hit_at_node);
        }
        // Snapshot child ids into the scratch buffer (not a fresh Vec). Borrow
        // of the children map is dropped before we recurse, so `&self` calls
        // below are free to walk the tree.
        let saved_stack_len = scratch.len();
        scratch.extend(self.tree.node(node_id).children.values().copied());
        // Drain only the portion we just added, in LIFO order, without reallocating.
        while scratch.len() > saved_stack_len {
            let child_id = scratch.pop().unwrap();
            let child_pc = self.tree.node(child_id).pending_count;
            if established && child_pc < min_pending_count {
                // Safe skip: main_hit already established for this path, and
                // this subtree doesn't meet the sharing threshold.
                self.assign_subtree_main_hit(child_id, main_hit_at_node, result, scratch);
                continue;
            }
            let edge_len = self.tree.node(child_id).edge.len();
            path.extend_from_slice(&self.tree.node(child_id).edge);
            let new_depth = path.len();
            // Build the PyList directly from path (no Vec<Token> clone first).
            let path_arg = PyList::new_bound(py, path.iter().copied());
            let match_len: usize = cache_match_fn.call1((path_arg,))?.extract()?;
            if match_len >= new_depth {
                self.dualwalk(
                    child_id,
                    new_depth,
                    true,
                    path,
                    result,
                    scratch,
                    py,
                    cache_match_fn,
                    min_pending_count,
                )?;
            } else {
                self.assign_subtree_main_hit(child_id, match_len, result, scratch);
            }
            path.truncate(path.len() - edge_len);
        }
        Ok(())
    }

    /// DFS assigns `main_hit` to every terminator in the subtree rooted at
    /// `root_id`. Uses `scratch` as its working stack; grows it, drains it
    /// back to the original length before returning (so the caller's own
    /// stack contents above `scratch.len()` at call time are untouched).
    fn assign_subtree_main_hit(
        &self,
        root_id: NodeId,
        main_hit: usize,
        result: &mut HashMap<Rid, usize>,
        scratch: &mut Vec<NodeId>,
    ) {
        let base = scratch.len();
        scratch.push(root_id);
        while scratch.len() > base {
            let node_id = scratch.pop().unwrap();
            let node = self.tree.node(node_id);
            for &rid in &node.terminators {
                result.insert(rid, main_hit);
            }
            scratch.extend(node.children.values().copied());
        }
    }
}
