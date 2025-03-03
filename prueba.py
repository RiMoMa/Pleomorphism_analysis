from stardist.models import StarDist2D
#rom stardist.utils import normalize
from csbdeep.utils import normalize
import numpy as np

model = StarDist2D.from_pretrained("2D_versatile_he")

# Prueba con una imagen pequeña aleatoria
test_image = np.random.rand(256, 256)
labels, _ = model.predict_instances(normalize(test_image))
print("Segmentación exitosa")

import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100, 100)
plt.imshow(x)
plt.show()
