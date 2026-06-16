//! Minimal float linear algebra. Matrices are row-major (m[row][col]).

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
    #[inline]
    pub const fn splat(s: f32) -> Self {
        Self { x: s, y: s, z: s }
    }
    #[inline]
    pub fn add(self, b: Vec3) -> Vec3 {
        Vec3::new(self.x + b.x, self.y + b.y, self.z + b.z)
    }
    #[inline]
    pub fn sub(self, b: Vec3) -> Vec3 {
        Vec3::new(self.x - b.x, self.y - b.y, self.z - b.z)
    }
    #[inline]
    pub fn scale(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    #[inline]
    pub fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
    #[inline]
    pub fn dot(self, b: Vec3) -> f32 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }
    #[inline]
    pub fn cross(self, b: Vec3) -> Vec3 {
        Vec3::new(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }
    #[inline]
    pub fn squared_norm(self) -> f32 {
        self.dot(self)
    }
    #[inline]
    pub fn norm(self) -> f32 {
        self.squared_norm().sqrt()
    }
    #[inline]
    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n <= 1e-20 {
            return self;
        }
        self.scale(1.0 / n)
    }
    #[inline]
    pub fn cwise_product(self, b: Vec3) -> Vec3 {
        Vec3::new(self.x * b.x, self.y * b.y, self.z * b.z)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    #[inline]
    pub const fn from_vec3(v: Vec3, w: f32) -> Self {
        Self { x: v.x, y: v.y, z: v.z, w }
    }
    #[inline]
    pub fn head3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    #[inline]
    pub fn add(self, b: Vec4) -> Vec4 {
        Vec4::new(self.x + b.x, self.y + b.y, self.z + b.z, self.w + b.w)
    }
    #[inline]
    pub fn sub(self, b: Vec4) -> Vec4 {
        Vec4::new(self.x - b.x, self.y - b.y, self.z - b.z, self.w - b.w)
    }
    #[inline]
    pub fn scale(self, s: f32) -> Vec4 {
        Vec4::new(self.x * s, self.y * s, self.z * s, self.w * s)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Mat3 {
    pub m: [[f32; 3]; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self { m: [[0.0; 3]; 3] }
    }
}

impl Mat3 {
    #[inline]
    pub fn mul_vec3(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Mat4 {
    pub m: [[f32; 4]; 4],
}

impl Default for Mat4 {
    fn default() -> Self {
        Self { m: [[0.0; 4]; 4] }
    }
}

impl Mat4 {
    #[inline]
    pub fn zero() -> Mat4 {
        Mat4::default()
    }
    #[inline]
    pub fn identity() -> Mat4 {
        let mut r = Mat4::default();
        r.m[0][0] = 1.0;
        r.m[1][1] = 1.0;
        r.m[2][2] = 1.0;
        r.m[3][3] = 1.0;
        r
    }
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.m[row][col]
    }
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, v: f32) {
        self.m[row][col] = v;
    }

    /// Row-combination GEMM form (no horizontal adds, SIMD-friendly).
    pub fn mul(&self, b: &Mat4) -> Mat4 {
        let mut r = Mat4::default();
        for i in 0..4 {
            for j in 0..4 {
                r.m[i][j] = self.m[i][0] * b.m[0][j]
                    + self.m[i][1] * b.m[1][j]
                    + self.m[i][2] * b.m[2][j]
                    + self.m[i][3] * b.m[3][j];
            }
        }
        r
    }

    #[inline]
    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z + self.m[0][3] * v.w,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z + self.m[1][3] * v.w,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z + self.m[2][3] * v.w,
            self.m[3][0] * v.x + self.m[3][1] * v.y + self.m[3][2] * v.z + self.m[3][3] * v.w,
        )
    }

    #[inline]
    pub fn block33(&self) -> Mat3 {
        Mat3 {
            m: [
                [self.m[0][0], self.m[0][1], self.m[0][2]],
                [self.m[1][0], self.m[1][1], self.m[1][2]],
                [self.m[2][0], self.m[2][1], self.m[2][2]],
            ],
        }
    }

    /// General 4x4 inverse (cofactor expansion).
    pub fn inverse(&self) -> Mat4 {
        let x = &self.m;
        let m = [
            x[0][0], x[0][1], x[0][2], x[0][3], x[1][0], x[1][1], x[1][2], x[1][3], x[2][0],
            x[2][1], x[2][2], x[2][3], x[3][0], x[3][1], x[3][2], x[3][3],
        ];
        let mut inv = [0.0f32; 16];
        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
            + m[9] * m[7] * m[14]
            + m[13] * m[6] * m[11]
            - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
            - m[8] * m[7] * m[14]
            - m[12] * m[6] * m[11]
            + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
            + m[8] * m[7] * m[13]
            + m[12] * m[5] * m[11]
            - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
            - m[8] * m[6] * m[13]
            - m[12] * m[5] * m[10]
            + m[12] * m[6] * m[9];
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
            - m[9] * m[3] * m[14]
            - m[13] * m[2] * m[11]
            + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
            + m[8] * m[3] * m[14]
            + m[12] * m[2] * m[11]
            - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
            - m[8] * m[3] * m[13]
            - m[12] * m[1] * m[11]
            + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
            + m[8] * m[2] * m[13]
            + m[12] * m[1] * m[10]
            - m[12] * m[2] * m[9];
        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
            + m[5] * m[3] * m[14]
            + m[13] * m[2] * m[7]
            - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
            - m[4] * m[3] * m[14]
            - m[12] * m[2] * m[7]
            + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
            + m[4] * m[3] * m[13]
            + m[12] * m[1] * m[7]
            - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
            - m[4] * m[2] * m[13]
            - m[12] * m[1] * m[6]
            + m[12] * m[2] * m[5];
        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
            - m[5] * m[3] * m[10]
            - m[9] * m[2] * m[7]
            + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
            + m[4] * m[3] * m[10]
            + m[8] * m[2] * m[7]
            - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
            - m[4] * m[3] * m[9]
            - m[8] * m[1] * m[7]
            + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
            + m[4] * m[2] * m[9]
            + m[8] * m[1] * m[6]
            - m[8] * m[2] * m[5];

        let mut det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if det == 0.0 {
            return Mat4::identity();
        }
        det = 1.0 / det;
        let mut r = Mat4::default();
        for i in 0..4 {
            for j in 0..4 {
                r.m[i][j] = inv[i * 4 + j] * det;
            }
        }
        r
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }
}

/// Shortest-arc rotation mapping unit vector a onto unit vector b.
pub fn quat_from_two_vectors(a_in: Vec3, b_in: Vec3) -> Quat {
    let a = a_in.normalized();
    let b = b_in.normalized();
    let mut c = a.dot(b);
    if c < -1.0 + 1e-6 {
        let mut axis = Vec3::new(1.0, 0.0, 0.0).cross(a);
        if axis.squared_norm() < 1e-6 {
            axis = Vec3::new(0.0, 1.0, 0.0).cross(a);
        }
        axis = axis.normalized();
        return Quat { x: axis.x, y: axis.y, z: axis.z, w: 0.0 };
    }
    let axis = a.cross(b);
    if c > 1.0 {
        c = 1.0;
    }
    let s = ((1.0 + c) * 2.0).sqrt();
    let inv_s = 1.0 / s;
    Quat { x: axis.x * inv_s, y: axis.y * inv_s, z: axis.z * inv_s, w: s * 0.5 }
}
