def getMatrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
  # Convert data to radians
  rotation = math.pi * rotation / 180.0
  shear = math.pi * shear / 180.0

  def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

  # Rotation matrix
  c1   = tf.math.cos(rotation)
  s1   = tf.math.sin(rotation)
  one  = tf.constant([1],dtype='float32')
  zero = tf.constant([0],dtype='float32')

  rotation_matrix = get_3x3_mat([c1,   s1,   zero,
                                  -s1,  c1,   zero,
                                  zero, zero, one])
  # Shear matrix
  c2 = tf.math.cos(shear)
  s2 = tf.math.sin(shear)

  shear_matrix = get_3x3_mat([one,  s2,   zero,
                              zero, c2,   zero,
                              zero, zero, one])
  # Zoom matrix
  zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero,
                              zero,            one/width_zoom, zero,
                              zero,            zero,           one])
  # Shift matrix
  shift_matrix = get_3x3_mat([one,  zero, height_shift,
                              zero, one,  width_shift,
                              zero, zero, one])

  # This will generate dot prouct of all 4 of transformations
  return K.dot(K.dot(rotation_matrix, shear_matrix),
                K.dot(zoom_matrix,     shift_matrix))

def transform(image, label=None):
  # For data configuration
  read_size = 256
  rotation=180.0
  shear=2.0
  height_zoom=8.0
  width_zoom=8.0
  height_shift=8.0
  width_shift=8.0
  # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
  # output - image randomly rotated, sheared, zoomed, and shifted
  DIM = read_size
  XDIM = DIM%2 #fix for size 331

  rot = rotation * tf.random.normal([1], dtype='float32')
  shr = shear * tf.random.normal([1], dtype='float32')
  h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / height_zoom
  w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / width_zoom
  h_shift = height_shift * tf.random.normal([1], dtype='float32')
  w_shift = width_zoom * tf.random.normal([1], dtype='float32')

  # GET TRANSFORMATION MATRIX
  m = getMatrix(rot,shr,h_zoom,w_zoom,h_shift,w_shift)

  # LIST DESTINATION PIXEL INDICES
  x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
  y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
  z   = tf.ones([DIM*DIM], dtype='int32')
  idx = tf.stack( [x,y,z] )

  # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
  idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
  idx2 = K.cast(idx2, dtype='int32')
  idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)

  # FIND ORIGIN PIXEL VALUES
  idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
  d    = tf.gather_nd(image, tf.transpose(idx3))

  if label is not None: return tf.reshape(d,[DIM, DIM,3]), label
  else: return tf.reshape(d,[DIM, DIM,3])

# For extra augmentation
def augmented(image, label=None):
  # For random flipping left to right
  image = tf.image.random_flip_left_right(image)
  # For random hue values
  image = tf.image.random_hue(image, 0.01)
  # For ranndom saturationn
  image = tf.image.random_saturation(image, 0.7, 1.3)
  # For adding random contrast
  image = tf.image.random_contrast(image, 0.8, 1.2)
  # For adding random brightness
  image = tf.image.random_brightness(image, 0.1)
  # If images and labels are given
  if label is not None: return image, label
  # If only image is given, typically given for test dataset
  else: return image
