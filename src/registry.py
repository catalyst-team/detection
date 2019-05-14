from catalyst.utils.registry import Registry

BACKBONES = Registry("backbone")


def _backbone_late_add(r: Registry):
    from . import backbone as m
    r.add_from_module(m)


BACKBONES.late_add(_backbone_late_add)
Backbone = BACKBONES.add


NECKS = Registry("neck")


def _neck_late_add(r: Registry):
    from . import neck as m
    r.add_from_module(m)


NECKS.late_add(_neck_late_add)
Neck = NECKS.add


HEADS = Registry("head")


def _head_late_add(r: Registry):
    from . import head as m
    r.add_from_module(m)


HEADS.late_add(_head_late_add)
Head = HEADS.add


DETECTORS = Registry("detector")


def _detector_late_add(r: Registry):
    from . import detector as m
    r.add_from_module(m)


DETECTORS.late_add(_detector_late_add)
Detector = DETECTORS.add
