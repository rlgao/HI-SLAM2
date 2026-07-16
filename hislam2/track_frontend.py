import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


def resolve_scal3r_frontend_prior_policy(config):
    scal3r_config = config.get("scal3r_prior", {}) or {}
    legacy_active = bool(config.get("use_scal3r_prior", scal3r_config.get("active", False)))

    online_config = config.get("online_scal3r", {}) or {}
    online_active = bool(online_config.get("active", False)) and bool(
        online_config.get(
            "consume_frontend_prior",
            online_config.get("use_frontend_prior", False),
        )
    )
    effective_config = dict(scal3r_config)
    if online_active:
        effective_config.update(online_config.get("frontend_prior", {}) or {})

    return {
        "active": online_active or legacy_active,
        "online": online_active,
        "min_confidence": float(effective_config.get("min_confidence", 0.0)),
        "min_confident_pixels": int(effective_config.get("min_confident_pixels", 128)),
        "initialize_dscale": str(effective_config.get("initialize_dscale", "median_ratio")),
    }


class TrackFrontend:
    def __init__(self, net, video, config, scal3r_prior_policy=None):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48)

        # local optimization window
        self.t1 = 0

        # frontent variables
        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2
        self.warmup = 12

        self.frontend_nms = config["frontend_nms"]
        self.keyframe_thresh = config["keyframe_thresh"]
        self.frontend_window = config["frontend_window"]
        self.frontend_thresh = config["frontend_thresh"]
        self.frontend_radius = config["frontend_radius"]
        self.video.mono_depth_alpha = config["mono_depth_alpha"]
        prior_policy = scal3r_prior_policy or resolve_scal3r_frontend_prior_policy(config)
        self.use_scal3r_prior = bool(prior_policy["active"])
        self.min_prior_confidence = float(prior_policy["min_confidence"])
        self.min_confident_pixels = int(prior_policy["min_confident_pixels"])
        self.initialize_dscale = str(prior_policy["initialize_dscale"])
        if self.initialize_dscale != "median_ratio":
            raise ValueError("scal3r_prior.initialize_dscale currently supports only 'median_ratio'")

    def _initialize_dscale(self, index):
        if not self.use_scal3r_prior:
            self.video.dscales[index] = self.video.disps[index].median() / self.video.disps_prior[index].median()
            return

        valid = (self.video.disps_prior[index] > 0) & (
            self.video.disps_prior_conf[index] > self.min_prior_confidence
        )
        if valid.sum().item() >= self.min_confident_pixels:
            disp_median = self.video.disps[index][valid].median()
            prior_median = self.video.disps_prior[index][valid].median()
            if torch.isfinite(disp_median).item() and torch.isfinite(prior_median).item() and prior_median.item() > 0:
                self.video.dscales[index] = disp_median / prior_median
                return

        self.video.dscales[index] = self.video.disps[index].median() / self.video.disps_prior[index].median()

    def __update(self, is_last):
        """ add edges, perform update """

        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, remove=True)

        self._initialize_dscale(self.t1-1)
        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True, use_mono=itr>1)

        d = self.video.distance([self.t1-3], [self.t1-2], bidirectional=True)
        d_covis = self.video.distance_covis([self.t1-2])
        covis_thresh = 0.1
        cri1 = d.item() < self.keyframe_thresh
        cri2 = d_covis.item() < covis_thresh
        if cri1 and cri2 and not is_last:
            self.graph.rm_keyframe(self.t1 - 2)
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
            update_idx = []
        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

            if is_last:
                update_idx = torch.arange(self.graph.ii.min(), self.t1, device='cuda')
            else:
                update_idx = torch.arange(self.graph.ii.min(), self.t1-1, device='cuda')

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True
        return update_idx

    def __initialize(self):
        """ initialize the SLAM system """

        self.t1 = self.video.counter.value

        # initial optimization
        self.graph.add_neighborhood_factors(0, self.t1, r=3)
        for itr in range(8):
            self.graph.update(1, use_inactive=True, use_mono=False)

        # refine optimization
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
        for i in range(self.t1):
            self._initialize_dscale(i)
        for itr in range(8):
            self.graph.update(1, use_inactive=True, use_mono=itr>2)

        # remove keyframes with too small motion
        while self.t1 > self.warmup-4:
            d = self.video.distance(torch.arange(0, self.t1-2), torch.arange(2, self.t1), bidirectional=True)
            if d.min() < self.keyframe_thresh:
                self.video.shift(d.argmin()+2, n=-1)
                self.t1 -= 1
            else:
                break

        # last optimization after removing too close keyframes
        self.graph.rm_factors(self.graph.ii > -1)
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
        for itr in range(8):
            self.graph.update(1, use_inactive=True, use_mono=itr>2)
        self.video.normalize()

        # initialization complete
        self.video.is_initialized = True
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()
        with self.video.get_lock():
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.t1-4, store=True)
        return torch.arange(self.t1-1, device='cuda')

    def __call__(self, is_last):
        """ main update """
        self.to_update = []

        # do initialization
        if not self.video.is_initialized and self.video.counter.value == self.warmup:
            self.to_update = self.__initialize()
            
        # do update
        elif self.video.is_initialized and self.t1 < self.video.counter.value:
            self.to_update = self.__update(is_last)
        
        return self.to_update
