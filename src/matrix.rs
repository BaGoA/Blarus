use std::ops::{Index, IndexMut};

/// The way how matrix data are stored.
/// Row major order stores matrix data row by row in contiguous memory vector
/// Column major order stores matrix data column by colunm in contiguous memory vector
enum StorageOrder {
    RowMajor,
    ColumnMajor,
}

/// Matrix elements accessor
/// The matrix is stored in contiguous memory vector. The accessor defines how we access to matrix element in this vector.
/// It contains strides along row and column that we need to apply to matrix indexes (i, j)
/// to obtain the memory location in vector.
/// There is also offset, if we want start to explore matrix from other index than (0, 0)
#[derive(Clone, Copy)]
pub struct Accessor {
    stride_row: usize,
    stride_col: usize,
    offset: usize,
}

impl Accessor {
    /// Create an accesor from stride along row and column
    /// We keep the offset to 0
    pub fn new(stride_row: usize, stride_col: usize) -> Self {
        return Self {
            stride_row,
            stride_col,
            offset: 0,
        };
    }

    /// Create an accessor from stride and offset along row and column
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

/// View on part of matrix, so it does not own data.
/// It contains number of rows and number of columns of view, an accessor.
pub struct View<'a, T> {
    nb_rows: usize,
    nb_cols: usize,
    accessor: Accessor,
    data: &'a [T],
}

impl<'a, T> View<'a, T> {
    /// Create a view from number of rows, number of columns, an accessor and a mutable slice
    pub fn new(nb_rows: usize, nb_cols: usize, accessor: Accessor, data: &'a [T]) -> Self {
        return Self {
            nb_rows,
            nb_cols,
            accessor,
            data,
        };
    }

    /// Get number of rows of view
    pub fn nb_rows(&self) -> usize {
        return self.nb_rows;
    }

    /// Get number of columns of view
    pub fn nb_cols(&self) -> usize {
        return self.nb_cols;
    }
}

impl<'a, T> Index<(usize, usize)> for View<'a, T> {
    type Output = T;

    /// This allows to read the view element at (index of row, index of column) position
    /// like this let element: f32 = view[(0, 2)];
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let id: usize = self.accessor.index(index.0, index.1);
        return self.data.index(id);
    }
}

/// Mutable view on part of matrix, so it does not own data.
/// It contains number of rows and number of columns of view, an accessor.
pub struct ViewMut<'a, T> {
    nb_rows: usize,
    nb_cols: usize,
    accessor: Accessor,
    data: &'a mut [T],
}

impl<'a, T> ViewMut<'a, T> {
    /// Create a mutable view from number of rows, number of columns, an accessor and a mutable slice
    pub fn new(nb_rows: usize, nb_cols: usize, accessor: Accessor, data: &'a mut [T]) -> Self {
        return Self {
            nb_rows,
            nb_cols,
            accessor,
            data,
        };
    }

    /// Get number of rows of mutable view
    pub fn nb_rows(&self) -> usize {
        return self.nb_rows;
    }

    /// Get number of columns of mutable view
    pub fn nb_cols(&self) -> usize {
        return self.nb_cols;
    }
}

impl<'a, T> Index<(usize, usize)> for ViewMut<'a, T> {
    type Output = T;

    /// This allows to read the view element at (index of row, index of column) position
    /// like this let element: f32 = view[(0, 2)];
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let id: usize = self.accessor.index(index.0, index.1);
        return self.data.index(id);
    }
}

impl<'a, T> IndexMut<(usize, usize)> for ViewMut<'a, T> {
    /// This allows to write an value in matrix at (index of row, index of column) position
    /// like this matrix[(0, 2)] = 3.1415;
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let id: usize = self.accessor.index(index.0, index.1);
        return self.data.index_mut(id);
    }
}

/// Matrix
/// This structure contains number of rows and number of columns of matrix, an accessor
/// to get memory position of elements in contiguous memory vector and vector to store matrix data
pub struct Matrix<T> {
    nb_rows: usize,
    nb_cols: usize,
    accessor: Accessor,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Default,
{
    // Create a row-major matrix from number of rows and columns of matrix
    pub fn new_row_major(nb_rows: usize, nb_cols: usize) -> Self {
        let mut data: Vec<T> = Vec::new();
        data.resize_with(nb_rows * nb_cols, Default::default);

        return Self {
            nb_rows,
            nb_cols,
            accessor: Accessor::new(nb_cols, 1),
            data,
        };
    }

    // Create a column-major matrix from number of rows and columns of matrix
    pub fn new_column_major(nb_rows: usize, nb_cols: usize) -> Self {
        let mut data: Vec<T> = Vec::new();
        data.resize_with(nb_rows * nb_cols, Default::default);

        return Self {
            nb_rows,
            nb_cols,
            accessor: Accessor::new(1, nb_rows),
            data,
        };
    }
}

/// View parameters
/// This structure contains this indexes of first element of view
/// and number of rows and number of colunm that we want
pub struct ViewParameters {
    start_row: usize,
    start_col: usize,
    nb_rows: usize,
    nb_cols: usize,
}

impl ViewParameters {
    pub fn new(start_row: usize, start_col: usize, nb_rows: usize, nb_cols: usize) -> Self {
        return ViewParameters {
            start_row,
            start_col,
            nb_rows,
            nb_cols,
        };
    }
}

impl<'a, T> Matrix<T> {
    /// Get full view of matrix
    pub fn full_view(&'a self) -> View<'a, T> {
        return View::new(
            self.nb_rows,
            self.nb_cols,
            self.accessor,
            self.data.as_slice(),
        );
    }

    /// Get full mutable view of matrix
    pub fn full_view_mut(&'a mut self) -> ViewMut<'a, T> {
        return ViewMut::new(
            self.nb_rows,
            self.nb_cols,
            self.accessor,
            self.data.as_mut_slice(),
        );
    }

    /// Get view on part of matrix
    pub fn view(&'a self, params: ViewParameters) -> View<'a, T> {
        return View::new(
            params.nb_rows,
            params.nb_cols,
            Accessor::new_with_offset(
                self.accessor.stride_row,
                self.accessor.stride_col,
                params.start_row,
                params.start_col,
            ),
            self.data.as_slice(),
        );
    }

    /// Get mutable view on part of matrix
    pub fn view_mut(&'a mut self, params: ViewParameters) -> ViewMut<'a, T> {
        return ViewMut::new(
            params.nb_rows,
            params.nb_cols,
            Accessor::new_with_offset(
                self.accessor.stride_row,
                self.accessor.stride_col,
                params.start_row,
                params.start_col,
            ),
            self.data.as_mut_slice(),
        );
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

    #[test]
    fn test_matrix_new_row_major() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 4;

        let matrix: Matrix<i32> = Matrix::new_row_major(nb_rows, nb_cols);

        assert_eq!(matrix.nb_rows, nb_rows);
        assert_eq!(matrix.nb_cols, nb_cols);
        assert_eq!(matrix.data.len(), nb_rows * nb_cols);
    }

    #[test]
    fn test_matrix_new_column_major() {
        let nb_rows: usize = 4;
        let nb_cols: usize = 3;

        let matrix: Matrix<i32> = Matrix::new_column_major(nb_rows, nb_cols);

        assert_eq!(matrix.nb_rows, nb_rows);
        assert_eq!(matrix.nb_cols, nb_cols);
        assert_eq!(matrix.data.len(), nb_rows * nb_cols);
    }

    #[test]
    fn test_matrix_dimensions_access() {
        let nb_rows: usize = 5;
        let nb_cols: usize = 3;

        let matrix: Matrix<i32> = Matrix::new_row_major(nb_rows, nb_cols);
        let matrix_view: View<i32> = matrix.full_view();

        assert_eq!(matrix_view.nb_rows(), nb_rows);
        assert_eq!(matrix_view.nb_cols(), nb_cols);
    }

    #[test]
    fn test_matrix_row_major_full_view() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let mut matrix: Matrix<i32> = Matrix::new_row_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        matrix.data = data_ref.clone();

        let view: View<i32> = matrix.full_view();

        assert_eq!(view[(0, 0)], data_ref[0]);
        assert_eq!(view[(0, 1)], data_ref[1]);
        assert_eq!(view[(0, 2)], data_ref[2]);
        assert_eq!(view[(1, 0)], data_ref[3]);
        assert_eq!(view[(1, 1)], data_ref[4]);
        assert_eq!(view[(1, 2)], data_ref[5]);
        assert_eq!(view[(2, 0)], data_ref[6]);
        assert_eq!(view[(2, 1)], data_ref[7]);
        assert_eq!(view[(2, 2)], data_ref[8]);
    }

    #[test]
    fn test_matrix_column_major_full_view() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let mut matrix: Matrix<i32> = Matrix::new_column_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        matrix.data = data_ref.clone();

        let view: View<i32> = matrix.full_view();

        assert_eq!(view[(0, 0)], data_ref[0]);
        assert_eq!(view[(1, 0)], data_ref[1]);
        assert_eq!(view[(2, 0)], data_ref[2]);
        assert_eq!(view[(0, 1)], data_ref[3]);
        assert_eq!(view[(1, 1)], data_ref[4]);
        assert_eq!(view[(2, 1)], data_ref[5]);
        assert_eq!(view[(0, 2)], data_ref[6]);
        assert_eq!(view[(1, 2)], data_ref[7]);
        assert_eq!(view[(2, 2)], data_ref[8]);
    }

    #[test]
    fn test_matrix_row_major_full_mutable_view() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let mut matrix: Matrix<i32> = Matrix::new_row_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        matrix.data = data_ref.clone();

        let factor: i32 = 3;

        {
            let mut view_mut: ViewMut<i32> = matrix.full_view_mut();

            view_mut[(1, 2)] *= factor;
            view_mut[(2, 1)] *= factor;
        }

        let view: View<i32> = matrix.full_view();

        assert_eq!(view[(0, 0)], data_ref[0]);
        assert_eq!(view[(0, 1)], data_ref[1]);
        assert_eq!(view[(0, 2)], data_ref[2]);
        assert_eq!(view[(1, 0)], data_ref[3]);
        assert_eq!(view[(1, 1)], data_ref[4]);
        assert_eq!(view[(1, 2)], factor * data_ref[5]);
        assert_eq!(view[(2, 0)], data_ref[6]);
        assert_eq!(view[(2, 1)], factor * data_ref[7]);
        assert_eq!(view[(2, 2)], data_ref[8]);
    }

    #[test]
    fn test_matrix_column_major_full_view_mut() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let mut matrix: Matrix<i32> = Matrix::new_column_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        matrix.data = data_ref.clone();

        let factor: i32 = 3;

        {
            let mut view_mut: ViewMut<i32> = matrix.full_view_mut();

            view_mut[(1, 0)] *= factor;
            view_mut[(2, 1)] *= factor;
        }

        let view: View<i32> = matrix.full_view();

        assert_eq!(view[(0, 0)], data_ref[0]);
        assert_eq!(view[(1, 0)], factor * data_ref[1]);
        assert_eq!(view[(2, 0)], data_ref[2]);
        assert_eq!(view[(0, 1)], data_ref[3]);
        assert_eq!(view[(1, 1)], data_ref[4]);
        assert_eq!(view[(2, 1)], factor * data_ref[5]);
        assert_eq!(view[(0, 2)], data_ref[6]);
        assert_eq!(view[(1, 2)], data_ref[7]);
        assert_eq!(view[(2, 2)], data_ref[8]);
    }

    #[test]
    fn test_matrix_row_major_view() {
        let nb_rows: usize = 4;
        let nb_cols: usize = 4;

        let mut matrix: Matrix<i32> = Matrix::new_row_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        matrix.data = data_ref.clone();

        let view: View<i32> = matrix.view(ViewParameters::new(1, 1, 2, 2));

        assert_eq!(view[(0, 0)], data_ref[5]);
        assert_eq!(view[(0, 1)], data_ref[6]);
        assert_eq!(view[(1, 0)], data_ref[9]);
        assert_eq!(view[(1, 1)], data_ref[10]);
    }

    #[test]
    fn test_matrix_column_major_view() {
        let nb_rows: usize = 4;
        let nb_cols: usize = 4;

        let mut matrix: Matrix<i32> = Matrix::new_column_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        matrix.data = data_ref.clone();

        let view: View<i32> = matrix.view(ViewParameters::new(1, 1, 2, 2));

        assert_eq!(view[(0, 0)], data_ref[5]);
        assert_eq!(view[(0, 1)], data_ref[9]);
        assert_eq!(view[(1, 0)], data_ref[6]);
        assert_eq!(view[(1, 1)], data_ref[10]);
    }

    #[test]
    fn test_matrix_row_major_view_mut() {
        let nb_rows: usize = 4;
        let nb_cols: usize = 4;

        let mut matrix: Matrix<i32> = Matrix::new_row_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        matrix.data = data_ref.clone();

        let factor: i32 = 3;

        {
            let mut view_mut: ViewMut<i32> = matrix.view_mut(ViewParameters::new(1, 1, 2, 2));

            view_mut[(0, 0)] *= factor;
            view_mut[(0, 1)] *= factor;
            view_mut[(1, 0)] *= factor;
            view_mut[(1, 1)] *= factor;
        }

        let view: View<i32> = matrix.full_view();

        assert_eq!(view[(0, 0)], data_ref[0]);
        assert_eq!(view[(0, 1)], data_ref[1]);
        assert_eq!(view[(0, 2)], data_ref[2]);
        assert_eq!(view[(0, 3)], data_ref[3]);
        assert_eq!(view[(1, 0)], data_ref[4]);
        assert_eq!(view[(1, 1)], factor * data_ref[5]);
        assert_eq!(view[(1, 2)], factor * data_ref[6]);
        assert_eq!(view[(1, 3)], data_ref[7]);
        assert_eq!(view[(2, 0)], data_ref[8]);
        assert_eq!(view[(2, 1)], factor * data_ref[9]);
        assert_eq!(view[(2, 2)], factor * data_ref[10]);
        assert_eq!(view[(2, 3)], data_ref[11]);
        assert_eq!(view[(3, 0)], data_ref[12]);
        assert_eq!(view[(3, 1)], data_ref[13]);
        assert_eq!(view[(3, 2)], data_ref[14]);
        assert_eq!(view[(3, 3)], data_ref[15]);
    }

    #[test]
    fn test_matrix_column_major_view_mut() {
        let nb_rows: usize = 4;
        let nb_cols: usize = 4;

        let mut matrix: Matrix<i32> = Matrix::new_column_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        matrix.data = data_ref.clone();

        let factor: i32 = 3;

        {
            let mut view_mut: ViewMut<i32> = matrix.view_mut(ViewParameters::new(1, 1, 2, 2));

            view_mut[(0, 0)] *= factor;
            view_mut[(0, 1)] *= factor;
            view_mut[(1, 0)] *= factor;
            view_mut[(1, 1)] *= factor;
        }

        let view: View<i32> = matrix.full_view();

        assert_eq!(view[(0, 0)], data_ref[0]);
        assert_eq!(view[(1, 0)], data_ref[1]);
        assert_eq!(view[(2, 0)], data_ref[2]);
        assert_eq!(view[(3, 0)], data_ref[3]);
        assert_eq!(view[(0, 1)], data_ref[4]);
        assert_eq!(view[(1, 1)], factor * data_ref[5]);
        assert_eq!(view[(2, 1)], factor * data_ref[6]);
        assert_eq!(view[(3, 1)], data_ref[7]);
        assert_eq!(view[(0, 2)], data_ref[8]);
        assert_eq!(view[(1, 2)], factor * data_ref[9]);
        assert_eq!(view[(2, 2)], factor * data_ref[10]);
        assert_eq!(view[(3, 2)], data_ref[11]);
        assert_eq!(view[(0, 3)], data_ref[12]);
        assert_eq!(view[(1, 3)], data_ref[13]);
        assert_eq!(view[(2, 3)], data_ref[14]);
        assert_eq!(view[(3, 3)], data_ref[15]);
    }
}
