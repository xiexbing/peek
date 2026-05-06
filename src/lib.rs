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

mod tree;
mod pending;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::pending::{PyPendingTree, lpm_sort_order, peek_clpm_sort_order};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPendingTree>()?;
    m.add_function(wrap_pyfunction!(lpm_sort_order, m)?)?;
    m.add_function(wrap_pyfunction!(peek_clpm_sort_order, m)?)?;
    Ok(())
}
