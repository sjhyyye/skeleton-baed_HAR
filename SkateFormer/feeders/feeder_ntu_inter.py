import numpy as np
import random

from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False, joint_indices=None, num_people=2):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.keep_joint_indices = joint_indices
        self.num_people = 2
        self.out_num_people = int(num_people)
        if self.out_num_people not in (1, 2):
            raise ValueError(f'num_people must be 1 or 2, got {self.out_num_people}')
        self.load_data()
        self._init_joint_indices()

    def _init_joint_indices(self):
        if self.keep_joint_indices is not None:
            joint_indices = np.array(self.keep_joint_indices, dtype=np.int64).reshape(-1)
            if joint_indices.size == 0:
                raise ValueError('joint_indices cannot be empty')
            if joint_indices.min() >= 1 and joint_indices.max() <= self.num_joints:
                joint_indices = joint_indices - 1
            if joint_indices.min() < 0 or joint_indices.max() >= self.num_joints:
                raise ValueError(f'joint_indices out of range for V={self.num_joints}: {joint_indices.tolist()}')
            if len(np.unique(joint_indices)) != len(joint_indices):
                raise ValueError('joint_indices cannot contain duplicates')
            self.keep_joint_indices = joint_indices

        if self.partition:
            if self.num_joints != 25:
                raise ValueError(f'partition=True requires V=25 joints, got V={self.num_joints}.')
            right_arm = np.array([7, 8, 22, 23]) - 1
            left_arm = np.array([11, 12, 24, 25]) - 1
            right_leg = np.array([13, 14, 15, 16]) - 1
            left_leg = np.array([17, 18, 19, 20]) - 1
            h_torso = np.array([5, 9, 6, 10]) - 1
            w_torso = np.array([2, 3, 1, 4]) - 1
            self.joint_indices = np.concatenate((right_arm, left_arm, right_leg, left_leg, h_torso, w_torso), axis=-1)
        else:
            self.joint_indices = None

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            data = npz_data['x_train']
            label = np.where(npz_data['y_train'] > 0)[1]
            inter_idx = np.where(((label >= 49) & (label <= 59)) | ((label >= 105) & (label <= 119)))
            self.data = data[inter_idx]
            self.label = label[inter_idx]
            for i in range(len(self.label)):
                if (self.label[i] >= 49) & (self.label[i] <= 59):
                    self.label[i] = self.label[i] - 49
                else:
                    self.label[i] = self.label[i] - 105 + 11
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            data = npz_data['x_test']
            label = np.where(npz_data['y_test'] > 0)[1]
            inter_idx = np.where(((label >= 49) & (label <= 59)) | ((label >= 105) & (label <= 119)))
            self.data = data[inter_idx]
            self.label = label[inter_idx]
            for i in range(len(self.label)):
                if (self.label[i] >= 49) & (self.label[i] <= 59):
                    self.label[i] = self.label[i] - 49
                else:
                    self.label[i] = self.label[i] - 105 + 11
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, D = self.data.shape
        if D % (self.num_people * 3) != 0:
            raise ValueError(f'Invalid data shape {self.data.shape}; expected last dim divisible by {self.num_people * 3}')
        self.num_joints = D // (self.num_people * 3)
        self.data = self.data.reshape((N, T, self.num_people, self.num_joints, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        if self.out_num_people == 1:
            data_numpy = data_numpy[:, :, :, :1]
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        num_people = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)

        if self.uniform:
            data_numpy, index_t = tools.valid_crop_uniform(data_numpy, valid_frame_num, self.p_interval,
                                                           self.window_size, self.thres)
        else:
            data_numpy, index_t = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval,
                                                          self.window_size, self.thres)

        if self.split == 'train':
            # intra-instance augmentation
            p = np.random.rand(1)
            if p < self.intra_p:

                if 'a' in self.aug_method:
                    if data_numpy.shape[-1] == 2 and np.random.rand(1) < 0.5:
                        data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if 'b' in self.aug_method:
                    if num_people == 2:
                        if np.random.rand(1) < 0.5:
                            axis_next = np.random.randint(0, 1)
                            temp = data_numpy.copy()
                            C, T, V, M = data_numpy.shape
                            x_new = np.zeros((C, T, V))
                            temp[:, :, :, axis_next] = x_new
                            data_numpy = temp

                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method:
                    data_numpy, index_t = tools.temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)

            # inter-instance augmentation
            elif (p < (self.intra_p + self.inter_p)) & (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = self.data[adain_idx]
                data_adain = np.array(data_adain)
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                t_idx = np.round((index_t + 1) * f_num / 2).astype(np.int)
                data_adain = data_adain[:, t_idx]
                data_numpy = tools.skeleton_adain_bone_length(data_numpy, data_adain)

            else:
                data_numpy = data_numpy.copy()

        # modality
        if self.data_type == 'b':
            j2b = tools.joint2bone()
            data_numpy = j2b(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = tools.to_motion(data_numpy)
        elif self.data_type == 'bm':
            j2b = tools.joint2bone()
            data_numpy = j2b(data_numpy)
            data_numpy = tools.to_motion(data_numpy)
        else:
            data_numpy = data_numpy.copy()

        if self.partition:
            data_numpy = data_numpy[:, :, self.joint_indices]
            if self.keep_joint_indices is not None:
                keep_mask = np.isin(self.joint_indices, self.keep_joint_indices)
                data_numpy[:, :, ~keep_mask] = 0
        else:
            if self.keep_joint_indices is not None:
                data_numpy = data_numpy[:, :, self.keep_joint_indices]

        return data_numpy, index_t, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
