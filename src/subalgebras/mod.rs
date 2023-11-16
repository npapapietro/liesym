pub mod branching_rules;
pub mod groups;

pub use branching_rules::*;
pub use groups::*;

pub type AlgResult<T> = Result<T, &'static str>;
