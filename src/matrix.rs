use std::ops::{Index, IndexMut};

use super::view::{Accessor, View, ViewMut};

/// Matrix
/// This structure contains number of rows and number of columns of matrix, an accessor
/// to get memory position of elements and a vector to store matrix data
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

    /// Get number of rows
    pub fn nb_rows(&self) -> usize {
        return self.nb_rows;
    }

    /// Get number of columns
    pub fn nb_cols(&self) -> usize {
        return self.nb_cols;
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
            self.accessor.clone(),
            self.data.as_slice(),
        );
    }

    /// Get full mutable view of matrix
    pub fn full_view_mut(&'a mut self) -> ViewMut<'a, T> {
        return ViewMut::new(
            self.nb_rows,
            self.nb_cols,
            self.accessor.clone(),
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

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    /// This allows to read the matrix element at (index of row, index of column) position
    /// like this let element: f32 = matrix[(0, 2)];
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let id: usize = self.accessor.index(index.0, index.1);
        return self.data.index(id);
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    /// This allows to write an value in matrix at (index of row, index of column) position
    /// like this matrix[(0, 2)] = 3.1415;
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let id: usize = self.accessor.index(index.0, index.1);
        return self.data.index_mut(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert_eq!(matrix.nb_rows(), nb_rows);
        assert_eq!(matrix.nb_cols(), nb_cols);
    }

    #[test]
    fn test_matrix_row_major_data_access() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let mut matrix: Matrix<i32> = Matrix::new_row_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        matrix.data = data_ref.clone();

        assert_eq!(matrix[(0, 0)], data_ref[0]);
        assert_eq!(matrix[(0, 1)], data_ref[1]);
        assert_eq!(matrix[(0, 2)], data_ref[2]);
        assert_eq!(matrix[(1, 0)], data_ref[3]);
        assert_eq!(matrix[(1, 1)], data_ref[4]);
        assert_eq!(matrix[(1, 2)], data_ref[5]);
        assert_eq!(matrix[(2, 0)], data_ref[6]);
        assert_eq!(matrix[(2, 1)], data_ref[7]);
        assert_eq!(matrix[(2, 2)], data_ref[8]);

        matrix[(2, 1)] = 43;
        assert_eq!(matrix[(2, 1)], 43);
    }

    #[test]
    fn test_matrix_column_major_data_access() {
        let nb_rows: usize = 3;
        let nb_cols: usize = 3;

        let mut matrix: Matrix<i32> = Matrix::new_column_major(nb_rows, nb_cols);

        let data_ref: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        matrix.data = data_ref.clone();

        assert_eq!(matrix[(0, 0)], data_ref[0]);
        assert_eq!(matrix[(1, 0)], data_ref[1]);
        assert_eq!(matrix[(2, 0)], data_ref[2]);
        assert_eq!(matrix[(0, 1)], data_ref[3]);
        assert_eq!(matrix[(1, 1)], data_ref[4]);
        assert_eq!(matrix[(2, 1)], data_ref[5]);
        assert_eq!(matrix[(0, 2)], data_ref[6]);
        assert_eq!(matrix[(1, 2)], data_ref[7]);
        assert_eq!(matrix[(2, 2)], data_ref[8]);

        matrix[(2, 1)] = 43;
        assert_eq!(matrix[(2, 1)], 43);
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
            let mut view: ViewMut<i32> = matrix.full_view_mut();

            view[(1, 2)] *= factor;
            view[(2, 1)] *= factor;
        }

        assert_eq!(matrix[(0, 0)], data_ref[0]);
        assert_eq!(matrix[(0, 1)], data_ref[1]);
        assert_eq!(matrix[(0, 2)], data_ref[2]);
        assert_eq!(matrix[(1, 0)], data_ref[3]);
        assert_eq!(matrix[(1, 1)], data_ref[4]);
        assert_eq!(matrix[(1, 2)], factor * data_ref[5]);
        assert_eq!(matrix[(2, 0)], data_ref[6]);
        assert_eq!(matrix[(2, 1)], factor * data_ref[7]);
        assert_eq!(matrix[(2, 2)], data_ref[8]);
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
            let mut view: ViewMut<i32> = matrix.full_view_mut();

            view[(1, 0)] *= factor;
            view[(2, 1)] *= factor;
        }

        assert_eq!(matrix[(0, 0)], data_ref[0]);
        assert_eq!(matrix[(1, 0)], factor * data_ref[1]);
        assert_eq!(matrix[(2, 0)], data_ref[2]);
        assert_eq!(matrix[(0, 1)], data_ref[3]);
        assert_eq!(matrix[(1, 1)], data_ref[4]);
        assert_eq!(matrix[(2, 1)], factor * data_ref[5]);
        assert_eq!(matrix[(0, 2)], data_ref[6]);
        assert_eq!(matrix[(1, 2)], data_ref[7]);
        assert_eq!(matrix[(2, 2)], data_ref[8]);
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
            let mut view: ViewMut<i32> = matrix.view_mut(ViewParameters::new(1, 1, 2, 2));

            view[(0, 0)] *= factor;
            view[(0, 1)] *= factor;
            view[(1, 0)] *= factor;
            view[(1, 1)] *= factor;
        }

        assert_eq!(matrix[(0, 0)], data_ref[0]);
        assert_eq!(matrix[(0, 1)], data_ref[1]);
        assert_eq!(matrix[(0, 2)], data_ref[2]);
        assert_eq!(matrix[(0, 3)], data_ref[3]);
        assert_eq!(matrix[(1, 0)], data_ref[4]);
        assert_eq!(matrix[(1, 1)], factor * data_ref[5]);
        assert_eq!(matrix[(1, 2)], factor * data_ref[6]);
        assert_eq!(matrix[(1, 3)], data_ref[7]);
        assert_eq!(matrix[(2, 0)], data_ref[8]);
        assert_eq!(matrix[(2, 1)], factor * data_ref[9]);
        assert_eq!(matrix[(2, 2)], factor * data_ref[10]);
        assert_eq!(matrix[(2, 3)], data_ref[11]);
        assert_eq!(matrix[(3, 0)], data_ref[12]);
        assert_eq!(matrix[(3, 1)], data_ref[13]);
        assert_eq!(matrix[(3, 2)], data_ref[14]);
        assert_eq!(matrix[(3, 3)], data_ref[15]);
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
            let mut view: ViewMut<i32> = matrix.view_mut(ViewParameters::new(1, 1, 2, 2));

            view[(0, 0)] *= factor;
            view[(0, 1)] *= factor;
            view[(1, 0)] *= factor;
            view[(1, 1)] *= factor;
        }

        assert_eq!(matrix[(0, 0)], data_ref[0]);
        assert_eq!(matrix[(1, 0)], data_ref[1]);
        assert_eq!(matrix[(2, 0)], data_ref[2]);
        assert_eq!(matrix[(3, 0)], data_ref[3]);
        assert_eq!(matrix[(0, 1)], data_ref[4]);
        assert_eq!(matrix[(1, 1)], factor * data_ref[5]);
        assert_eq!(matrix[(2, 1)], factor * data_ref[6]);
        assert_eq!(matrix[(3, 1)], data_ref[7]);
        assert_eq!(matrix[(0, 2)], data_ref[8]);
        assert_eq!(matrix[(1, 2)], factor * data_ref[9]);
        assert_eq!(matrix[(2, 2)], factor * data_ref[10]);
        assert_eq!(matrix[(3, 2)], data_ref[11]);
        assert_eq!(matrix[(0, 3)], data_ref[12]);
        assert_eq!(matrix[(1, 3)], data_ref[13]);
        assert_eq!(matrix[(2, 3)], data_ref[14]);
        assert_eq!(matrix[(3, 3)], data_ref[15]);
    }
}
