// Stub constants for IDE (rust-analyzer) only. These mirror the defaults used in
// nlhe configuration so that the codebase type-checks when build scripts are not executed.

// NOTE: This file is **only** compiled when `cfg(rust_analyzer)` is active.

// This file is included at the crate level and defines the associated constants
// inside an `impl Odds { ... }` block. It is only compiled when the `rust_analyzer`
// cfg is active (i.e. during IDE analysis).

impl Odds {
    pub const GRID: [Self; 10] = Self::PREF_RAISES;

    pub const PREF_RAISES: [Self; 10] = [
        Self(1, 4), // 0.25
        Self(1, 3), // 0.33
        Self(1, 2), // 0.50
        Self(2, 3), // 0.66
        Self(3, 4), // 0.75
        Self(1, 1), // 1.00
        Self(3, 2), // 1.50
        Self(2, 1), // 2.00
        Self(3, 1), // 3.00
        Self(4, 1), // 4.00
    ];

    pub const FLOP_RAISES: [Self; 5] = [
        Self(1, 2), // 0.50
        Self(3, 4), // 0.75
        Self(1, 1), // 1.00
        Self(3, 2), // 1.50
        Self(2, 1), // 2.00
    ];

    pub const LATE_RAISES: [Self; 4] = [
        Self(1, 2), // 0.50
        Self(1, 1), // 1.00
        Self(3, 2), // 1.50
        Self(2, 1), // 2.00
    ];

    pub const LAST_RAISES: [Self; 2] = [
        Self(1, 1), // 1.00
        Self(3, 2), // 1.50
    ];
} 