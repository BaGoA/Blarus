/// This structure define how we access to memory location from matrix indexes (i, j).
/// It contains strides along row and column that we need to apply to matrix indexes (i, j)
/// to obtain the memory location in vector which store matrix data.
/// There is also offset, if we want start to explore matrix from other index than (0, 0)
pub struct Accessor {
    stride_row: usize,
    stride_col: usize,
    offset: usize,
}

impl Accessor {
    // Constructor from stride along row and column
    // We keep the offset to 0
    pub fn new(stride_row: usize, stride_col: usize) -> Self {
        return Self {
            stride_row,
            stride_col,
            offset: 0,
        };
    }

    // Constructor from stride and offset along row and column
    pub fn new_with_offset(
        stride_row: usize,
        stride_col: usize,
        offset_row: usize,
        offset_col: usize,
    ) -> Self {
        let offset: usize = stride_row * offset_row + stride_col * offset_col;

        return Self {
            stride_row,
            stride_col,
            offset,
        };
    }

    /// Compute memory location in vector from row index and colunm index
    pub fn index(&self, row_id: usize, col_id: usize) -> usize {
        return row_id * self.stride_row + col_id * self.stride_col + self.offset;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accessor_new() {
        let stride_row: usize = 2;
        let stride_col: usize = 3;

        let accessor = Accessor::new(stride_row, stride_col);
        assert_eq!(accessor.stride_row, stride_row);
        assert_eq!(accessor.stride_col, stride_col);
        assert_eq!(accessor.offset, 0);
    }

    #[test]
    fn test_accessor_new_with_offset() {
        let stride_row: usize = 2;
        let stride_col: usize = 3;
        let offset_row: usize = 1;
        let offset_col: usize = 1;

        let accessor = Accessor::new_with_offset(stride_row, stride_col, offset_row, offset_col);
        assert_eq!(accessor.stride_row, stride_row);
        assert_eq!(accessor.stride_col, stride_col);

        let offset_ref: usize = stride_row * offset_row + stride_col * offset_col;
        assert_eq!(accessor.offset, offset_ref);
    }

    #[test]
    fn test_accessor_index() {
        let stride_row: usize = 3;
        let stride_col: usize = 3;

        let mut accessor = Accessor::new(stride_row, 1);
        assert_eq!(accessor.index(1, 2), stride_row + 2);

        accessor = Accessor::new(1, stride_col);
        assert_eq!(accessor.index(2, 1), 2 + stride_col);
    }

    #[test]
    fn test_accessor_index_with_offset() {
        let stride_row: usize = 4;
        let stride_col: usize = 4;
        let offset_row: usize = 1;
        let offset_col: usize = 1;

        let mut accessor = Accessor::new_with_offset(stride_row, 1, offset_row, offset_col);
        assert_eq!(accessor.index(1, 2), stride_row + 7);

        accessor = Accessor::new_with_offset(1, stride_col, offset_row, offset_col);
        assert_eq!(accessor.index(2, 1), 7 + stride_col);
    }
}
