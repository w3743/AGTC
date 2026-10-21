"""Map raw 4K video coordinates to the calibrated, upright depth grid."""
import cv2
import numpy as np
from realtime import _box_candidates


class Range4K:
    def __init__(self, calibration):
        with np.load(calibration, allow_pickle=False) as values:
            self.k, self.d, self.r, self.p = [values[name].copy() for name in ('K1','D1','R1','P1')]
        self.inverse_projection = np.linalg.inv(self.p[:,:3])

    @staticmethod
    def scale_crop(shape):
        height, width = shape
        scale = height / 3840
        # Infer the crop from the actual shape (not a rounded scale label).
        crop = (2160 * scale - width) / 2
        return scale, crop

    def depth_pixel(self, x, y, shape):
        if not np.isfinite([x,y]).all() or not (0<=x<3840 and 0<=y<2160):
            raise ValueError('4K pixel outside the camera image')
        point = cv2.undistortPoints(np.array([[[x,y]]],np.float64),self.k,self.d,
                                   R=self.r,P=self.p).reshape(2)
        scale,crop = self.scale_crop(shape)
        px = point[1]*scale+(scale-1)/2-crop
        py = shape[0]-1-(point[0]*scale+(scale-1)/2)
        return int(round(px)), int(round(py))

    def raw_pixels(self, points, shape):
        scale,crop = self.scale_crop(shape)
        points=np.asarray(points,np.float64)
        raw_rect=np.column_stack(((shape[0]-1-points[:,1]-(scale-1)/2)/scale,
                                  (points[:,0]+crop-(scale-1)/2)/scale,
                                  np.ones(len(points))))
        rays=(self.r.T @ self.inverse_projection @ raw_rect.T).T
        projected,_=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),self.k,self.d)
        return projected.reshape(-1,2)

    def boxes(self, depth):
        output=[]
        for box in _box_candidates(depth):
            x,y,w,h=box['x'],box['y'],box['width'],box['height']
            corners=self.raw_pixels([(x,y),(x+w-1,y),(x+w-1,y+h-1),(x,y+h-1)],depth.shape)
            output.append(dict(depth_m=round(box['depth_m'],3),polygon=corners.round(2).tolist()))
        return output
