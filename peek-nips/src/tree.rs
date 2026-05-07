// Copyright 2026 Anonymous Authors
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

//! Arena-backed radix tree over token-ID sequences.
//!
//! Nodes live in a `Vec<Node>` indexed by `NodeId`. The root has id 0 and an
//! empty edge. Each non-root node's `edge` holds the token sequence consumed on
//! the edge from its parent. `children` is keyed by the first token of the
//! child's edge (so lookup at a branching point is O(1) on first token).
//!
//! `pending_count` is the number of pending requests whose path passes through
//! this node. `terminators` is the set of request IDs whose path *ends* at this
//! node. Together they let `longest_shared_prefix(rid)` answer "deepest node on
//! rid's path shared with at least one other pending request" in O(depth).

use rustc_hash::{FxHashMap, FxHashSet};

pub type Token = u32;
pub type Rid = u64;
pub type NodeId = u32;

pub const ROOT: NodeId = 0;

/// Count how many leading tokens of `edge` equal those of `tokens`.
#[inline]
fn match_edge_len(edge: &[Token], tokens: &[Token]) -> usize {
    edge.iter()
        .zip(tokens.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

#[derive(Debug)]
pub struct Node {
    pub edge: Vec<Token>,
    pub parent: NodeId,
    pub children: FxHashMap<Token, NodeId>,
    pub pending_count: u32,
    pub terminators: FxHashSet<Rid>,
    /// # of completions recorded at this node (running sample count).
    pub decode_samples: u32,
    /// Exponentially-weighted running mean of actual output-token length for
    /// reqs whose full token path ended at this node. Zero if no samples.
    pub decode_ewma: f32,
}

impl Node {
    fn new(edge: Vec<Token>, parent: NodeId) -> Self {
        Self {
            edge,
            parent,
            children: FxHashMap::default(),
            pending_count: 0,
            terminators: FxHashSet::default(),
            decode_samples: 0,
            decode_ewma: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct Tree {
    pub nodes: Vec<Node>,
    /// Freelist of dead slots; reused before pushing.
    free: Vec<NodeId>,
}

impl Tree {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::new(Vec::new(), ROOT)],
            free: Vec::new(),
        }
    }

    fn alloc(&mut self, edge: Vec<Token>, parent: NodeId) -> NodeId {
        if let Some(id) = self.free.pop() {
            self.nodes[id as usize] = Node::new(edge, parent);
            id
        } else {
            let id = self.nodes.len() as NodeId;
            self.nodes.push(Node::new(edge, parent));
            id
        }
    }

    fn free(&mut self, id: NodeId) {
        debug_assert!(id != ROOT);
        // Clear to drop large allocations eagerly; slot is reused via freelist.
        self.nodes[id as usize] = Node::new(Vec::new(), ROOT);
        self.free.push(id);
    }

    #[inline]
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    #[inline]
    fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    /// Split `node` at `offset` within its edge. After the split, the upper
    /// half keeps `node`'s id (preserves parent's child pointer) and a new
    /// node holding the lower half + the original children/terminators is
    /// created as its single child. `node`'s id continues to refer to the
    /// upper half.
    fn split_edge(&mut self, node: NodeId, offset: usize) {
        debug_assert!(node != ROOT);
        let n = self.node_mut(node);
        debug_assert!(offset > 0 && offset < n.edge.len());

        let lower_edge: Vec<Token> = n.edge.split_off(offset);
        let lower_first = lower_edge[0];

        // Move children, terminators, pending_count, decode stats to lower half.
        let children = std::mem::take(&mut n.children);
        let terminators = std::mem::take(&mut n.terminators);
        let pending_count = n.pending_count;
        let decode_samples = n.decode_samples;
        let decode_ewma = n.decode_ewma;
        n.decode_samples = 0;
        n.decode_ewma = 0.0;

        let lower = self.alloc(lower_edge, node);
        // Reparent grandchildren while we still own `children` locally (no
        // conflicting borrow of the tree through `lower`).
        for &g in children.values() {
            self.node_mut(g).parent = lower;
        }
        {
            let l = self.node_mut(lower);
            l.children = children;
            l.terminators = terminators;
            l.pending_count = pending_count;
            l.decode_samples = decode_samples;
            l.decode_ewma = decode_ewma;
        }
        self.node_mut(node).children.insert(lower_first, lower);
    }

    /// Descend from `node` consuming tokens[i..]. Returns (end_node, end_offset,
    /// consumed) where `end_offset` is the offset within `end_node.edge` at
    /// which the descent stopped (equal to edge.len() if we stopped at a
    /// node boundary) and `consumed` is the number of tokens matched total.
    /// Stops at the first mismatch or when tokens are exhausted.
    pub fn descend(&self, tokens: &[Token]) -> Descent {
        let mut node = ROOT;
        let mut consumed = 0usize;
        loop {
            if consumed == tokens.len() {
                return Descent {
                    end_node: node,
                    end_offset: self.node(node).edge.len(),
                    consumed,
                };
            }
            let next_tok = tokens[consumed];
            let Some(&child) = self.node(node).children.get(&next_tok) else {
                return Descent {
                    end_node: node,
                    end_offset: self.node(node).edge.len(),
                    consumed,
                };
            };
            let edge = &self.node(child).edge;
            // Match along the edge.
            let common = match_edge_len(edge, &tokens[consumed..]);
            consumed += common;
            if common < edge.len() {
                // Stopped partway along the edge.
                return Descent {
                    end_node: child,
                    end_offset: common,
                    consumed,
                };
            }
            // Consumed the whole edge, continue from child.
            node = child;
        }
    }

    /// Insert a terminator for `rid` at the path defined by `tokens`. Splits
    /// edges / creates nodes as needed. Increments `pending_count` on every
    /// node along the resulting path (root excluded). Idempotent: inserting
    /// the same rid twice with the same tokens is a no-op at the terminator
    /// level but will still double-count `pending_count`, so callers must
    /// ensure each rid is inserted once.
    pub fn insert(&mut self, rid: Rid, tokens: &[Token]) {
        let d = self.descend(tokens);
        let terminator = if d.consumed == tokens.len() {
            // Reached end of tokens.
            if d.end_offset != self.node(d.end_node).edge.len() {
                self.split_edge(d.end_node, d.end_offset);
            }
            d.end_node
        } else {
            // Mismatch partway: possibly split, then add a new child holding the tail.
            if d.end_offset != self.node(d.end_node).edge.len() {
                self.split_edge(d.end_node, d.end_offset);
            }
            let branch = d.end_node;
            let tail: Vec<Token> = tokens[d.consumed..].to_vec();
            let first = tail[0];
            let new = self.alloc(tail, branch);
            self.node_mut(branch).children.insert(first, new);
            new
        };

        self.node_mut(terminator).terminators.insert(rid);
        // Bump pending_count on every node from terminator up to (but not including) root.
        let mut cur = terminator;
        while cur != ROOT {
            self.node_mut(cur).pending_count += 1;
            cur = self.node(cur).parent;
        }
    }

    /// Remove `rid` from the tree. Returns true if the rid was present. After
    /// removal, garbage-collects empty branches and merges any parent that
    /// ends up with exactly one child and no terminators.
    pub fn remove(&mut self, rid: Rid, tokens: &[Token]) -> bool {
        let d = self.descend(tokens);
        if d.consumed != tokens.len()
            || d.end_offset != self.node(d.end_node).edge.len()
        {
            return false;
        }
        let terminator = d.end_node;
        if !self.node_mut(terminator).terminators.remove(&rid) {
            return false;
        }
        // Decrement pending_count along the path.
        let mut cur = terminator;
        while cur != ROOT {
            self.node_mut(cur).pending_count -= 1;
            cur = self.node(cur).parent;
        }
        // GC: if terminator has no terminators and no children, delete it; then
        // walk upward, merging single-child chains with empty terminators.
        self.gc(terminator);
        true
    }

    fn gc(&mut self, mut node: NodeId) {
        while node != ROOT {
            let n = self.node(node);
            // Keep nodes that hold decode-prediction history even if they
            // have no live terminators/children — they carry per-cluster
            // EWMA used at admission time to estimate decode length.
            if n.terminators.is_empty()
                && n.children.is_empty()
                && n.decode_samples == 0
            {
                let parent = n.parent;
                let first = n.edge[0];
                self.node_mut(parent).children.remove(&first);
                self.free(node);
                node = parent;
                continue;
            }
            // Try merging: node has no terminators and exactly one child → fuse edges.
            // Skip merging if either node has decode history (merging would lose the
            // boundary between two distinct cluster prefixes).
            if n.terminators.is_empty() && n.children.len() == 1 && node != ROOT && n.decode_samples == 0 {
                let child = *n.children.values().next().unwrap();
                // Merge child's edge into node, then promote child's children/terminators.
                let child_edge = std::mem::take(&mut self.node_mut(child).edge);
                let child_children = std::mem::take(&mut self.node_mut(child).children);
                let child_terms = std::mem::take(&mut self.node_mut(child).terminators);
                self.node_mut(node).edge.extend(child_edge);
                // Reparent grandchildren while we still own `child_children`
                // locally (no conflicting borrow through `node`).
                for &g in child_children.values() {
                    self.node_mut(g).parent = node;
                }
                self.node_mut(node).children = child_children;
                self.node_mut(node).terminators = child_terms;
                self.free(child);
                // Continue upward: the merged node may itself now be mergeable.
                // But it has terminators/children from child, so further merging
                // is unlikely; stop regardless — parent is unaffected by this merge.
                return;
            }
            return;
        }
    }

    /// Walk `tokens` and return the length of the longest prefix whose path
    /// passes through nodes with `pending_count >= threshold`. Mid-edge matches
    /// are counted: along any edge from P to C, the count is uniformly
    /// C.pending_count (no branches exist mid-edge by radix tree compression),
    /// so a partial-edge match can safely claim the count of the edge's child.
    /// This matches sglang's RadixCache.match_prefix semantics (which reports
    /// the true prefix length even when it ends inside an edge).
    pub fn deepest_with_count(&self, tokens: &[Token], threshold: u32) -> usize {
        let mut node = ROOT;
        let mut consumed = 0usize;
        let mut best = 0usize;
        loop {
            if consumed == tokens.len() {
                return best;
            }
            let next_tok = tokens[consumed];
            let Some(&child) = self.node(node).children.get(&next_tok) else {
                return best;
            };
            let edge = &self.node(child).edge;
            let common = match_edge_len(edge, &tokens[consumed..]);
            consumed += common;
            if self.node(child).pending_count >= threshold {
                best = consumed;
            }
            if common < edge.len() {
                return best;
            }
            node = child;
        }
    }

    /// Record an observed completion: walk `tokens` into the tree (creating/
    /// splitting edges as needed so a terminal node sits exactly at
    /// `tokens.len()`), then update that node's EWMA with `output_len`.
    ///
    /// The tree must outlive the original pending entry. `remove(rid)` leaves
    /// nodes with recorded decode data alive (see `gc`) specifically so this
    /// history can be queried at later admission time.
    pub fn record_decode(&mut self, tokens: &[Token], output_len: u32) {
        // Use the same path-materialization logic as `insert` (without
        // touching pending_count or terminators): walk edges, splitting and
        // creating as needed until a node sits at depth tokens.len().
        let mut node = ROOT;
        let mut consumed = 0usize;
        while consumed < tokens.len() {
            let next_tok = tokens[consumed];
            if let Some(&child) = self.node(node).children.get(&next_tok) {
                let edge_len = self.node(child).edge.len();
                let remaining = &tokens[consumed..];
                let common = match_edge_len(&self.node(child).edge, remaining);
                if common == edge_len {
                    consumed += common;
                    node = child;
                    continue;
                }
                // Tokens diverge from this edge mid-way → split so a terminal
                // node sits at the divergence point, then either follow into
                // the new branch or create a fresh leaf.
                self.split_edge(child, common);
                consumed += common;
                node = child;
                if consumed == tokens.len() {
                    break;
                }
                // Create a new branch for the remaining tokens.
                let remainder = tokens[consumed..].to_vec();
                let rem_len = remainder.len();
                let first = remainder[0];
                let leaf = self.alloc(remainder, node);
                self.node_mut(node).children.insert(first, leaf);
                node = leaf;
                consumed += rem_len;
                break;
            }
            // No matching child → create a fresh leaf for remaining tokens.
            let remainder = tokens[consumed..].to_vec();
            let rem_len = remainder.len();
            let first = remainder[0];
            let leaf = self.alloc(remainder, node);
            self.node_mut(node).children.insert(first, leaf);
            node = leaf;
            consumed += rem_len;
            break;
        }
        // Update EWMA at the terminal node.
        const ALPHA: f32 = 0.3;
        let n = self.node_mut(node);
        if n.decode_samples == 0 {
            n.decode_ewma = output_len as f32;
        } else {
            n.decode_ewma = ALPHA * (output_len as f32) + (1.0 - ALPHA) * n.decode_ewma;
        }
        n.decode_samples = n.decode_samples.saturating_add(1);
    }

    /// Predict decode length for a new req with token path `tokens`.
    ///
    /// Walks `tokens` into the tree and returns the EWMA recorded at the
    /// deepest node with at least `min_samples` observations. Walks from
    /// deep → shallow so the most specific cluster wins; falls back to
    /// shallower ancestors when deeper nodes haven't seen enough completions.
    ///
    /// Returns `Some((ewma, samples))` on hit, `None` if even root has no
    /// qualifying ancestor. Mid-edge queries inherit the deeper node's stats
    /// — any node at or below the query position shares the same path prefix.
    pub fn predict_decode(&self, tokens: &[Token], min_samples: u32) -> Option<(f32, u32)> {
        // Walk into the tree, remembering every node whose decode_samples
        // meets the threshold. The deepest such node is the prediction.
        let mut node = ROOT;
        let mut consumed = 0usize;
        let mut best: Option<(f32, u32)> = None;
        loop {
            let n = self.node(node);
            if n.decode_samples >= min_samples {
                best = Some((n.decode_ewma, n.decode_samples));
            }
            if consumed == tokens.len() {
                return best;
            }
            let next_tok = tokens[consumed];
            let Some(&child) = n.children.get(&next_tok) else {
                return best;
            };
            let edge = &self.node(child).edge;
            let common = match_edge_len(edge, &tokens[consumed..]);
            if common < edge.len() {
                // Mid-edge: inherits child's stats.
                if self.node(child).decode_samples >= min_samples {
                    best = Some((
                        self.node(child).decode_ewma,
                        self.node(child).decode_samples,
                    ));
                }
                return best;
            }
            consumed += common;
            node = child;
        }
    }

    /// Total node count (for introspection/debugging).
    pub fn len(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    /// Fast "is there any sharing among pending rids?" check. O(#root-children).
    ///
    /// When this returns false, every pending rid is a singleton — peek's
    /// dualwalk / cluster_info can't produce any main_hit better than what
    /// a per-rid cache query would get. Callers should bypass peek scheduling
    /// and fall back to stock LPM in that case.
    pub fn has_sharing(&self) -> bool {
        self.node(ROOT)
            .children
            .values()
            .any(|&c| self.node(c).pending_count >= 2)
    }

    /// Number of pending rids that pass through a root child with
    /// `pending_count >= 2` — i.e., rids that share at least their first
    /// edge with some other pending rid. O(#root-children).
    ///
    /// `shared + singleton == total pending` (by construction of the tree).
    pub fn shared_rid_count(&self) -> u32 {
        self.node(ROOT)
            .children
            .values()
            .map(|&c| {
                let pc = self.node(c).pending_count;
                if pc >= 2 { pc } else { 0 }
            })
            .sum()
    }

    /// Number of pending rids whose first edge is unique (no other pending
    /// rid shares even the first token). Complement of `shared_rid_count`.
    pub fn singleton_rid_count(&self) -> u32 {
        self.node(ROOT)
            .children
            .values()
            .map(|&c| {
                let pc = self.node(c).pending_count;
                if pc < 2 { pc } else { 0 }
            })
            .sum()
    }

    /// Expose internal nodes for snapshot dumps. Caller must filter via
    /// `free_slots()` to skip freed slots.
    pub fn nodes_slice(&self) -> &[Node] {
        &self.nodes
    }

    /// IDs of freelist slots that are currently garbage.
    pub fn free_slots(&self) -> &[NodeId] {
        &self.free
    }

    /// Count pending rids whose token sequence ENDS EXACTLY at `path` —
    /// i.e., terminators at the peek node sitting at this path.
    ///
    /// Returns 0 when `path` ends mid-edge in the peek tree (no node sits at
    /// that exact depth) or when the path diverges from the tree. Contrast
    /// with `pending_demand`, which counts the whole subtree.
    ///
    /// Motivation: used as the eviction-priority signal for cache *leaf*
    /// nodes, so a shared-prefix node is not protected N× just because N
    /// sessions fan out beneath it — each specific cache path gets weighted
    /// only by the number of pending rids whose exact tokens need it.
    pub fn terminators_at(&self, path: &[Token]) -> u32 {
        let d = self.descend(path);
        if d.consumed != path.len() {
            return 0;
        }
        if d.end_offset != self.node(d.end_node).edge.len() {
            // Path landed mid-edge → no peek node exists at this exact depth.
            return 0;
        }
        self.node(d.end_node).terminators.len() as u32
    }

    /// Count pending rids whose token sequence begins with `path`.
    ///
    /// Walks `path` through the tree, consuming it along edges. If `path`
    /// diverges from the tree before it's exhausted, returns 0 — no pending
    /// rid has this exact prefix. Otherwise returns the `pending_count` of
    /// the deepest reached node, which equals the number of terminators in
    /// its subtree (i.e., pending rids whose tokens start with `path`).
    ///
    /// Useful for demand-aware cache eviction: given a candidate cache node
    /// and its path, this tells you how many pending reqs would have to
    /// re-prefill that prefix if it gets evicted.
    pub fn pending_demand(&self, path: &[Token]) -> u32 {
        let mut node = ROOT;
        let mut consumed = 0usize;
        loop {
            if consumed == path.len() {
                return self.node(node).pending_count;
            }
            let next_tok = path[consumed];
            let Some(&child) = self.node(node).children.get(&next_tok) else {
                return 0;
            };
            let edge = &self.node(child).edge;
            let common = match_edge_len(edge, &path[consumed..]);
            consumed += common;
            if consumed == path.len() {
                // path exhausted within or at end of this edge: every rid in
                // child's subtree starts with `path`.
                return self.node(child).pending_count;
            }
            if common < edge.len() {
                // Diverged mid-edge — no rid has this prefix.
                return 0;
            }
            node = child;
        }
    }

    /// Walk `tokens` down the tree as deep as possible and return
    /// `(match_depth, Option<sample_rid>)` where `match_depth` is the number
    /// of tokens matched (longest prefix of `tokens` present in the tree),
    /// and `sample_rid` is any pending rid whose path shares that prefix.
    ///
    /// Used by peek's handoff mechanism: given a near-finishing running req's
    /// full token prefix, find the pending rid that will reuse its KV cache
    /// most. Pioneer-to-sibling handoff pre-positions this rid in the
    /// scheduling queue so it admits at the exact moment the running slot
    /// frees (zero-gap cache reuse).
    ///
    /// Returns `(0, None)` if no prefix of `tokens` matches any pending rid.
    /// When multiple rids share the same max-depth prefix (common for cluster
    /// pioneer-sibling), returns an arbitrary one; caller can pick among them
    /// by other criteria (e.g., arrival order).
    pub fn longest_match_along(&self, tokens: &[Token]) -> (u32, Option<Rid>) {
        let mut node = ROOT;
        let mut consumed = 0usize;
        loop {
            if consumed == tokens.len() {
                break;
            }
            let next_tok = tokens[consumed];
            let Some(&child) = self.node(node).children.get(&next_tok) else {
                break;
            };
            let edge = &self.node(child).edge;
            let common = match_edge_len(edge, &tokens[consumed..]);
            consumed += common;
            if common < edge.len() {
                // Partial match into child's edge. Rids in child's subtree
                // share our first `consumed` tokens (they all have edge's
                // tokens beyond that, which diverge from ours). Return a
                // sample from child's subtree at depth = consumed.
                let rid = self.sample_terminator_in_subtree(child);
                return (consumed as u32, rid);
            }
            node = child;
        }
        if consumed == 0 {
            return (0, None);
        }
        let rid = self.sample_terminator_in_subtree(node);
        (consumed as u32, rid)
    }

    /// Internal helper: return any terminator rid found in the subtree rooted
    /// at `node` (including `node` itself). Prefers direct terminators, then
    /// descends into children. Returns None if the subtree has no terminators.
    fn sample_terminator_in_subtree(&self, node: NodeId) -> Option<Rid> {
        let n = self.node(node);
        if let Some(&r) = n.terminators.iter().next() {
            return Some(r);
        }
        for &child in n.children.values() {
            if let Some(r) = self.sample_terminator_in_subtree(child) {
                return Some(r);
            }
        }
        None
    }

    /// For a pending rid whose path is `tokens`, find the deepest ancestor of
    /// its terminator node with `pending_count >= 2`. That ancestor is the
    /// "cluster node" for `rid` — the finest-grained shared-prefix group it
    /// belongs to.
    ///
    /// Returns `(cluster_node_id, cluster_depth, cluster_size)`:
    ///   * `cluster_node_id` is an opaque identifier stable within a
    ///     mutation-free sequence of queries, suitable for grouping rids into
    ///     clusters.
    ///   * `cluster_depth` is the number of tokens in the shared prefix.
    ///   * `cluster_size` is the pending_count at the cluster node (number of
    ///     pending rids passing through it).
    ///
    /// Returns `None` if `rid` isn't present at `tokens`' terminator, or if
    /// the rid is a singleton (no ancestor — and not the terminator itself —
    /// has `pending_count >= 2`).
    pub fn cluster_info(
        &self,
        rid: Rid,
        tokens: &[Token],
    ) -> Option<(NodeId, usize, u32)> {
        let d = self.descend(tokens);
        if d.consumed != tokens.len()
            || d.end_offset != self.node(d.end_node).edge.len()
        {
            return None;
        }
        if !self.node(d.end_node).terminators.contains(&rid) {
            return None;
        }
        let mut cur = d.end_node;
        let mut depth_at_cur = tokens.len();
        while cur != ROOT {
            let n = self.node(cur);
            if n.pending_count >= 2 {
                return Some((cur, depth_at_cur, n.pending_count));
            }
            depth_at_cur -= n.edge.len();
            cur = n.parent;
        }
        None
    }
}

#[derive(Debug)]
pub struct Descent {
    pub end_node: NodeId,
    pub end_offset: usize,
    pub consumed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(xs: &[u32]) -> Vec<Token> {
        xs.to_vec()
    }

    #[test]
    fn insert_single() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3]));
        let d = t.descend(&toks(&[1, 2, 3]));
        assert_eq!(d.consumed, 3);
        assert_eq!(d.end_offset, t.node(d.end_node).edge.len());
        assert!(t.node(d.end_node).terminators.contains(&1));
    }

    #[test]
    fn insert_shared_prefix_splits_edge() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3, 4]));
        t.insert(2, &toks(&[1, 2, 5, 6]));
        // Root should have exactly one child (the shared prefix [1,2]).
        assert_eq!(t.node(ROOT).children.len(), 1);
        let (_, &c) = t.node(ROOT).children.iter().next().unwrap();
        assert_eq!(t.node(c).edge, toks(&[1, 2]));
        assert_eq!(t.node(c).pending_count, 2);
        assert_eq!(t.node(c).children.len(), 2);
    }

    #[test]
    fn insert_prefix_of_existing() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3, 4]));
        t.insert(2, &toks(&[1, 2])); // strict prefix
        let d = t.descend(&toks(&[1, 2]));
        assert_eq!(d.consumed, 2);
        assert!(t.node(d.end_node).terminators.contains(&2));
    }

    #[test]
    fn remove_restores_tree() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3, 4]));
        t.insert(2, &toks(&[1, 2, 5, 6]));
        let before = t.len();
        assert!(t.remove(2, &toks(&[1, 2, 5, 6])));
        // After removing rid 2, the split should collapse; only rid 1's path remains.
        let d = t.descend(&toks(&[1, 2, 3, 4]));
        assert_eq!(d.consumed, 4);
        assert!(t.node(d.end_node).terminators.contains(&1));
        // And the tree should have shrunk.
        assert!(t.len() < before);
        // Root → one child with edge [1,2,3,4].
        assert_eq!(t.node(ROOT).children.len(), 1);
        let (_, &c) = t.node(ROOT).children.iter().next().unwrap();
        assert_eq!(t.node(c).edge, toks(&[1, 2, 3, 4]));
    }

    #[test]
    fn remove_missing_returns_false() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2]));
        assert!(!t.remove(99, &toks(&[1, 2])));
        assert!(!t.remove(1, &toks(&[9, 9])));
    }

    #[test]
    fn deepest_with_count_finds_shared_prefix() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3, 4]));
        t.insert(2, &toks(&[1, 2, 5, 6]));
        // With threshold=2, deepest shared point between them is after [1,2].
        assert_eq!(t.deepest_with_count(&toks(&[1, 2, 3, 4]), 2), 2);
        assert_eq!(t.deepest_with_count(&toks(&[1, 2, 9, 9]), 2), 2);
        // No overlap.
        assert_eq!(t.deepest_with_count(&toks(&[7, 8, 9]), 2), 0);
    }

    #[test]
    fn deepest_with_count_reports_partial_edge() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3, 4]));
        t.insert(2, &toks(&[1, 2, 3, 4])); // same path, same edge
        // Along the single edge [1,2,3,4], pending_count is uniformly 2, so any
        // partial match is legitimate at threshold=2.
        assert_eq!(t.deepest_with_count(&toks(&[1, 2, 3]), 2), 3);
        assert_eq!(t.deepest_with_count(&toks(&[1, 2, 3, 4]), 2), 4);
    }

    #[test]
    fn pending_demand_exact_match() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3]));
        t.insert(2, &toks(&[1, 2, 3]));
        t.insert(3, &toks(&[1, 2, 9]));
        // Three rids share [1, 2]; only two terminate at [1, 2, 3].
        assert_eq!(t.pending_demand(&toks(&[1, 2])), 3);
        assert_eq!(t.pending_demand(&toks(&[1, 2, 3])), 2);
        assert_eq!(t.pending_demand(&toks(&[1, 2, 9])), 1);
    }

    #[test]
    fn pending_demand_diverged() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3]));
        assert_eq!(t.pending_demand(&toks(&[1, 9])), 0);
        assert_eq!(t.pending_demand(&toks(&[9, 9])), 0);
    }

    #[test]
    fn pending_demand_path_longer_than_any_rid() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3]));
        // Path [1,2,3,4] extends past rid 1's tokens — no rid has this prefix.
        assert_eq!(t.pending_demand(&toks(&[1, 2, 3, 4])), 0);
    }

    #[test]
    fn pending_demand_mid_edge() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3, 4, 5]));
        t.insert(2, &toks(&[1, 2, 3, 4, 6]));
        // Query mid-edge: [1,2,3] lands inside the shared edge [1,2,3,4].
        // Both rids extend this path → demand = 2.
        assert_eq!(t.pending_demand(&toks(&[1, 2, 3])), 2);
    }

    #[test]
    fn terminators_at_distinguishes_shared_ancestor_from_leaf() {
        let mut t = Tree::new();
        // Two rids share [1,2] but terminate at different leaves.
        t.insert(1, &toks(&[1, 2, 3]));
        t.insert(2, &toks(&[1, 2, 9]));
        // Shared ancestor has fan-out 2 but NO terminators exactly at [1,2].
        assert_eq!(t.terminators_at(&toks(&[1, 2])), 0);
        // Each leaf has exactly one terminator.
        assert_eq!(t.terminators_at(&toks(&[1, 2, 3])), 1);
        assert_eq!(t.terminators_at(&toks(&[1, 2, 9])), 1);
        // Mid-edge path has no node sitting there → 0.
        assert_eq!(t.terminators_at(&toks(&[1])), 0);
        // Duplicate terminators get counted.
        t.insert(3, &toks(&[1, 2, 3]));
        assert_eq!(t.terminators_at(&toks(&[1, 2, 3])), 2);
        // Diverged / too-long paths → 0.
        assert_eq!(t.terminators_at(&toks(&[1, 2, 3, 4])), 0);
        assert_eq!(t.terminators_at(&toks(&[7, 7])), 0);
    }

    #[test]
    fn cluster_info_singleton_returns_none() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3]));
        assert_eq!(t.cluster_info(1, &toks(&[1, 2, 3])), None);
    }

    #[test]
    fn cluster_info_two_siblings_share_at_branch() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3, 4]));
        t.insert(2, &toks(&[1, 2, 5, 6]));
        // Branch is at depth 2. Both rids in a cluster of size 2.
        let (n1, d1, s1) = t.cluster_info(1, &toks(&[1, 2, 3, 4])).unwrap();
        let (n2, d2, s2) = t.cluster_info(2, &toks(&[1, 2, 5, 6])).unwrap();
        assert_eq!(n1, n2);
        assert_eq!((d1, s1), (2, 2));
        assert_eq!((d2, s2), (2, 2));
    }

    #[test]
    fn cluster_info_three_way_with_nested_siblings() {
        let mut t = Tree::new();
        // Common prefix [1,2]. Under it: [3,4] shared by r1,r2; [5] for r3.
        // So r1,r2 form a tight cluster at depth 4; r3 clusters with them
        // only at depth 2 (the [1,2] node).
        t.insert(1, &toks(&[1, 2, 3, 4, 10]));
        t.insert(2, &toks(&[1, 2, 3, 4, 11]));
        t.insert(3, &toks(&[1, 2, 5]));
        let (n1, d1, s1) = t.cluster_info(1, &toks(&[1, 2, 3, 4, 10])).unwrap();
        let (n2, d2, s2) = t.cluster_info(2, &toks(&[1, 2, 3, 4, 11])).unwrap();
        let (n3, d3, s3) = t.cluster_info(3, &toks(&[1, 2, 5])).unwrap();
        assert_eq!(n1, n2, "r1 and r2 share finest cluster");
        assert_ne!(n1, n3, "r3 is in a coarser cluster");
        assert_eq!((d1, s1), (4, 2));
        assert_eq!((d3, s3), (2, 3));
    }

    #[test]
    fn cluster_info_duplicates_terminate_at_same_node() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[7, 8, 9]));
        t.insert(2, &toks(&[7, 8, 9]));
        let (n1, d1, s1) = t.cluster_info(1, &toks(&[7, 8, 9])).unwrap();
        let (n2, d2, s2) = t.cluster_info(2, &toks(&[7, 8, 9])).unwrap();
        assert_eq!(n1, n2);
        assert_eq!((d1, s1), (3, 2));
        assert_eq!((d2, s2), (3, 2));
    }

    #[test]
    fn sharing_introspection_four_cases() {
        // case 1: no sharing (all unique first tokens)
        let mut t = Tree::new();
        for (rid, first) in [(1u64, 10u32), (2, 20), (3, 30)] {
            t.insert(rid, &toks(&[first, first + 1, first + 2]));
        }
        assert!(!t.has_sharing());
        assert_eq!(t.shared_rid_count(), 0);
        assert_eq!(t.singleton_rid_count(), 3);

        // case 2: identical first token, diverge later — that IS sharing
        let mut t = Tree::new();
        t.insert(1, &toks(&[10, 20, 30]));
        t.insert(2, &toks(&[10, 40, 50]));
        assert!(t.has_sharing());
        assert_eq!(t.shared_rid_count(), 2);
        assert_eq!(t.singleton_rid_count(), 0);

        // case 3: mixed — one cluster of 3, one singleton with a different first token
        let mut t = Tree::new();
        t.insert(1, &toks(&[10, 20]));
        t.insert(2, &toks(&[10, 30]));
        t.insert(3, &toks(&[10, 40]));
        t.insert(4, &toks(&[99]));
        assert!(t.has_sharing());
        assert_eq!(t.shared_rid_count(), 3);
        assert_eq!(t.singleton_rid_count(), 1);

        // case 4: empty tree
        let t = Tree::new();
        assert!(!t.has_sharing());
        assert_eq!(t.shared_rid_count(), 0);
        assert_eq!(t.singleton_rid_count(), 0);
    }

    #[test]
    fn match_prefix_reports_partial_edge_single_rid() {
        let mut t = Tree::new();
        t.insert(1, &toks(&[1, 2, 3, 4]));
        // threshold=1 is the "match_prefix" semantic (any pending request).
        assert_eq!(t.deepest_with_count(&toks(&[1, 2, 3]), 1), 3);
        assert_eq!(t.deepest_with_count(&toks(&[1, 2]), 1), 2);
        assert_eq!(t.deepest_with_count(&toks(&[1, 9]), 1), 1);
        assert_eq!(t.deepest_with_count(&toks(&[9, 9]), 1), 0);
    }

    // ==========================================================================
    // Invariant-checked stress tests. Random insert/remove workloads that keep
    // a ground-truth set of (rid → tokens) outside the tree and re-derive every
    // queryable quantity from scratch, asserting parity with the tree.
    // ==========================================================================

    use std::collections::HashMap as StdHashMap;

    /// Deterministic tiny xorshift RNG — no dev-deps needed.
    struct XorShift(u64);
    impl XorShift {
        fn new(seed: u64) -> Self { Self(seed.max(1)) }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            self.0 = x; x
        }
        fn gen_range(&mut self, n: u64) -> u64 { self.next_u64() % n }
    }

    /// Ground-truth re-derive: walk the tree from root, sum pending_count from
    /// terminators-in-subtree; assert every node's stored pending_count matches.
    fn assert_tree_invariants(t: &Tree, ground: &StdHashMap<Rid, Vec<Token>>) {
        // 1. pending_count at every node equals the number of ground-truth rids
        //    whose tokens start with the path from root to that node.
        //    Re-derive paths by traversal.
        fn walk(t: &Tree, id: NodeId, path: &mut Vec<Token>, ground: &StdHashMap<Rid, Vec<Token>>) {
            let node = t.node(id);
            if id != ROOT {
                // pending_count: # ground rids whose tokens start with `path`.
                let expect = ground.values()
                    .filter(|toks| toks.len() >= path.len() && toks[..path.len()] == path[..])
                    .count() as u32;
                assert_eq!(node.pending_count, expect,
                    "pending_count mismatch at path {:?}: stored={}, expected={}",
                    path, node.pending_count, expect);
                // terminators: rids whose ground-truth tokens exactly equal `path`.
                let expect_term: std::collections::HashSet<Rid> = ground.iter()
                    .filter(|(_, toks)| toks.len() == path.len() && toks[..] == path[..])
                    .map(|(&r, _)| r).collect();
                let stored_term: std::collections::HashSet<Rid> = node.terminators.iter().copied().collect();
                assert_eq!(stored_term, expect_term,
                    "terminators mismatch at path {:?}: stored={:?}, expected={:?}",
                    path, stored_term, expect_term);
            }
            for &child_id in node.children.values() {
                let edge = &t.node(child_id).edge;
                let start = path.len();
                path.extend_from_slice(edge);
                walk(t, child_id, path, ground);
                path.truncate(start);
            }
        }
        let mut path = Vec::new();
        walk(t, ROOT, &mut path, ground);
    }

    fn random_tokens(rng: &mut XorShift, min_len: usize, max_len: usize, vocab: u32) -> Vec<Token> {
        let len = min_len + rng.gen_range((max_len - min_len) as u64 + 1) as usize;
        (0..len).map(|_| rng.gen_range(vocab as u64) as u32).collect()
    }

    #[test]
    fn stress_random_insert_remove_keeps_invariants() {
        // 2000 ops, small vocabulary so prefixes collide frequently.
        let mut t = Tree::new();
        let mut ground: StdHashMap<Rid, Vec<Token>> = StdHashMap::new();
        let mut rng = XorShift::new(0xC0FFEE);
        for _ in 0..2000 {
            let op = rng.gen_range(3);
            match op {
                0 => {
                    // Insert a new rid with random tokens (avoid rid collisions).
                    let rid = rng.next_u64() % 10_000;
                    if ground.contains_key(&rid) { continue; }
                    let toks = random_tokens(&mut rng, 0, 12, 5);
                    t.insert(rid, &toks);
                    ground.insert(rid, toks);
                }
                1 => {
                    // Remove an existing rid.
                    if ground.is_empty() { continue; }
                    let keys: Vec<Rid> = ground.keys().copied().collect();
                    let victim = keys[rng.gen_range(keys.len() as u64) as usize];
                    let toks = ground.remove(&victim).unwrap();
                    assert!(t.remove(victim, &toks));
                }
                _ => {
                    // Query pending_demand / terminators_at against ground truth.
                    // Note: empty-path queries are intentionally not supported by
                    // pending_demand — ROOT's pending_count is not maintained as
                    // an optimization (callers skip empty paths explicitly). We
                    // therefore restrict random queries to nonempty paths.
                    let q = random_tokens(&mut rng, 1, 6, 5);
                    let expected_demand = ground.values()
                        .filter(|v| v.len() >= q.len() && v[..q.len()] == q[..])
                        .count() as u32;
                    let got_demand = t.pending_demand(&q);
                    assert_eq!(got_demand, expected_demand,
                        "pending_demand mismatch at {:?}: got {}, expect {}", q, got_demand, expected_demand);
                    let expected_term = ground.values()
                        .filter(|v| v.len() == q.len() && v[..] == q[..])
                        .count() as u32;
                    let got_term = t.terminators_at(&q);
                    assert_eq!(got_term, expected_term,
                        "terminators_at mismatch at {:?}: got {}, expect {}", q, got_term, expected_term);
                }
            }
            assert_tree_invariants(&t, &ground);
        }
        // After all ops, final sweep.
        assert_tree_invariants(&t, &ground);
        // Drain everything — check GC brings tree back toward empty.
        let keys: Vec<Rid> = ground.keys().copied().collect();
        for k in keys {
            let toks = ground.remove(&k).unwrap();
            assert!(t.remove(k, &toks));
        }
        assert_tree_invariants(&t, &ground);
        // After full drain: only root remains (len == 1) with no children.
        assert_eq!(t.len(), 1, "tree not fully collapsed after draining all rids");
        assert!(t.node(ROOT).children.is_empty());
        assert_eq!(t.node(ROOT).pending_count, 0);
    }

    #[test]
    fn stress_long_paths_shared_prefixes() {
        // Agent-sessions-like: 10 agents × 50 sessions × up-to-5 turns.
        // SP is 50 tokens; per-session history accumulates.
        let mut t = Tree::new();
        let mut ground: StdHashMap<Rid, Vec<Token>> = StdHashMap::new();
        let mut rid_counter: Rid = 0;
        let mut rng = XorShift::new(0xB00B1E);
        for agent in 0..10u32 {
            let sp: Vec<Token> = (1000..1050).map(|i| i + agent * 100).collect();
            for _sess in 0..50 {
                let mut history = sp.clone();
                for _turn in 0..5 {
                    let mut toks = history.clone();
                    // Add a per-turn user-message "chunk" (5-15 random tokens).
                    toks.extend(random_tokens(&mut rng, 5, 15, 10_000));
                    let rid = rid_counter; rid_counter += 1;
                    t.insert(rid, &toks);
                    ground.insert(rid, toks.clone());
                    history = toks;
                }
            }
        }
        assert_tree_invariants(&t, &ground);
        // Random-delete half to stress GC + invariants.
        let keys: Vec<Rid> = ground.keys().copied().collect();
        for i in 0..keys.len()/2 {
            let k = keys[(i * 7) % keys.len()];
            if let Some(toks) = ground.remove(&k) {
                assert!(t.remove(k, &toks));
            }
        }
        assert_tree_invariants(&t, &ground);
    }

    #[test]
    fn stress_cluster_info_consistency() {
        // For any rid, cluster_info should report a (node, depth, size) where:
        //   - the node has pending_count == size >= 2
        //   - depth tokens of rid's path lead to that node
        //   - all rids reporting the same cluster node have identical depth/size
        let mut t = Tree::new();
        let mut ground: StdHashMap<Rid, Vec<Token>> = StdHashMap::new();
        let mut rng = XorShift::new(0xDEADBEEF);
        for rid in 0..500 {
            let toks = random_tokens(&mut rng, 1, 10, 4);
            t.insert(rid, &toks);
            ground.insert(rid, toks);
        }
        let mut per_cluster: StdHashMap<NodeId, Vec<(Rid, usize, u32)>> = StdHashMap::new();
        for (&rid, toks) in &ground {
            if let Some((nid, depth, size)) = t.cluster_info(rid, toks) {
                assert!(size >= 2, "cluster size must be >= 2");
                assert!(depth <= toks.len(), "cluster depth exceeds rid's tokens");
                per_cluster.entry(nid).or_default().push((rid, depth, size));
            }
        }
        // Invariant (per `cluster_info` docstring): size = pending_count at
        // the cluster node (= subtree count), NOT the number of rids whose
        // deepest cluster is this node. Rids with a deeper shared ancestor
        // will report that deeper ancestor instead, so members reporting X
        // may be a subset of size.
        for (nid, members) in &per_cluster {
            let (depth0, size0) = (members[0].1, members[0].2);
            for &(_rid, depth, size) in members {
                assert_eq!((depth, size), (depth0, size0),
                    "inconsistent (depth, size) within cluster {}", nid);
            }
            // size equals the stored pending_count on the node.
            assert_eq!(size0, t.node(*nid).pending_count,
                "cluster {}: reported size {} != node.pending_count {}",
                nid, size0, t.node(*nid).pending_count);
            // Every reporter's tokens must pass through this node. Re-derive
            // path to node by walking parents.
            let mut path_tokens: Vec<Token> = Vec::new();
            let mut cur = *nid;
            while cur != ROOT {
                let e = &t.node(cur).edge;
                let mut seg = e.clone();
                seg.extend(std::mem::take(&mut path_tokens));
                path_tokens = seg;
                cur = t.node(cur).parent;
            }
            // All `size0` rids whose tokens begin with `path_tokens` should exist
            // in ground; this subtree count must match.
            let subtree_count = ground.values()
                .filter(|v| v.len() >= path_tokens.len() && v[..path_tokens.len()] == path_tokens[..])
                .count() as u32;
            assert_eq!(size0, subtree_count,
                "cluster {}: reported size {} != ground subtree count {} (path {:?})",
                nid, size0, subtree_count, path_tokens);
        }
    }
}
