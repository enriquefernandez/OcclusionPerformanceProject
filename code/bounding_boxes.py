import numpy as np
import attr

@attr.s
class BoundingBox(object):
    xmin = attr.ib()
    ymin = attr.ib()
    xmax = attr.ib()
    ymax = attr.ib()

    def to_tuple(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


# Bounding box functions
def bounding_box_size(w,h,covered=0.5):
    scale = np.sqrt(covered)
    return int(scale*w), int(scale*h)


def center_bounding(bb,c=0.5):
    minx, miny, maxx, maxy = bb
    w, h = maxx-minx, maxy-miny
    nw,nh = bounding_box_size(maxx-minx, maxy-miny, covered=c)
    x_off, y_off = int((w-nw)/2), int((h-nh)/2)
    # print("w:{}, h:{}, nw:{}, nh:{}, x:{}, y:{}".
    #      format(w,h, nw, nh, x_off, y_off))

    # return minx+x_off, miny+y_off, minx+x_off+nw, miny+y_off+nh
    return BoundingBox(minx+x_off, miny+y_off, minx+x_off+nw, miny+y_off+nh)


def corner_bounding_box(corner_loc, bb, c=0.5):
    """corner_loc= 0 top left, 1 top right, 2 bottom right, 3 bottom left"""
    minx, miny, maxx, maxy = bb
    w, h = maxx - minx, maxy - miny
    nw, nh = bounding_box_size(maxx - minx, maxy - miny, covered=c)

    offsets = {0: (0,0), 1: (w-nw, 0), 2: (w-nw, h-nh), 3:(0, h-nh)}
    x_off, y_off = offsets[corner_loc]

    return BoundingBox(minx + x_off, miny + y_off, minx + x_off + nw, miny + y_off + nh)

def random_bounding_box(bb, c=0.5):
    minx, miny, maxx, maxy = bb
    w, h = maxx - minx, maxy - miny
    nw, nh = bounding_box_size(maxx - minx, maxy - miny, covered=c)

    x_off, y_off = np.random.randint(0, w-nw), np.random.randint(0, h - nh)

    return BoundingBox(minx + x_off, miny + y_off, minx + x_off + nw, miny + y_off + nh)