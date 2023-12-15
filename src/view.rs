use std::ops::Index;

/// Accessor
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

/// View
/// This struture is a view on part of matrix, so it does not own data.
struct View<'a, T> {
    nb_rows: usize,
    nb_cols: usize,
    accessor: Accessor,
    data: &'a [T],
}

impl<'a, T> View<'a, T> {
    pub fn new(nb_rows: usize, nb_cols: usize, accessor: Accessor, data: &'a [T]) -> Self {
        return Self {
            nb_rows,
            nb_cols,
            accessor,
            data,
        };
    }

    pub fn nb_rows(&self) -> usize {
        return self.nb_rows;
    }

    pub fn nb_cols(&self) -> usize {
        return self.nb_cols;
    }
}

impl<'a, T> Index<(usize, usize)> for View<'a, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let id: usize = self.accessor.index(index.0, index.1);
        return self.data.index(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

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

    #[test]
    fn test_view_new() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let view: View<i32> =
            View::new(nb_rows, nb_cols, Accessor::new(nb_cols, 1), data.as_slice());

        assert_eq!(view.nb_rows, nb_rows);
        assert_eq!(view.nb_cols, nb_cols);

        match view.data.partial_cmp(data.as_slice()) {
            Some(result) => assert_eq!(result, Ordering::Equal),
            None => assert!(false),
        }
    }

    #[test]
    fn test_view_dimensions_access() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let view: View<i32> =
            View::new(nb_rows, nb_cols, Accessor::new(nb_cols, 1), data.as_slice());

        assert_eq!(view.nb_rows(), nb_rows);
        assert_eq!(view.nb_cols(), nb_cols);
    }

    #[test]
    fn test_view_data_access() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let view: View<i32> =
            View::new(nb_rows, nb_cols, Accessor::new(nb_cols, 1), data.as_slice());

        assert_eq!(view[(0, 0)], data[0]);
        assert_eq!(view[(0, 1)], data[1]);
        assert_eq!(view[(0, 2)], data[2]);
        assert_eq!(view[(1, 0)], data[3]);
        assert_eq!(view[(1, 1)], data[4]);
        assert_eq!(view[(1, 2)], data[5]);
        assert_eq!(view[(2, 0)], data[6]);
        assert_eq!(view[(2, 1)], data[7]);
        assert_eq!(view[(2, 2)], data[8]);
    }

    #[test]
    fn test_view_data_access_with_offset() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let view: View<i32> = View::new(
            nb_rows - 1,
            nb_cols - 1,
            Accessor::new_with_offset(nb_cols, 1, 1, 1),
            data.as_slice(),
        );

        assert_eq!(view[(0, 0)], data[4]);
        assert_eq!(view[(0, 1)], data[5]);
        assert_eq!(view[(1, 0)], data[7]);
        assert_eq!(view[(1, 1)], data[8]);
    }
}
