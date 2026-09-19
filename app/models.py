#Imports
import torch
import torch.nn as nn
from app.ConfigManager import (NUM_CLASSES, DROPOUT, VOXEL_RESOLUTION, VOXEL_CHANNELS, VOXEL_FC,
                               MESH_HIDDEN, MESH_FC, NUM_POINTS, POINT_MLP, POINT_HEAD)


#3D CNN (voxels)
class VoxelCNN(nn.Module):    # PyTorch needs a class with __init__ and forward for every model
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        # each block: convolution, normalization, activation, then halve the grid size
        for out_channels in VOXEL_CHANNELS:
            layers = layers + [nn.Conv3d(in_channels, out_channels, 3, padding=1),
                               nn.BatchNorm3d(out_channels), nn.ReLU(), nn.MaxPool3d(2)]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        size = VOXEL_RESOLUTION // (2 ** len(VOXEL_CHANNELS))    # grid size left after all the pooling
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_channels * size ** 3, VOXEL_FC),
                                        nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(VOXEL_FC, NUM_CLASSES))

    def forward(self, x):    # x: (batch, 1, 32, 32, 32)
        return self.classifier(self.features(x))


#PointNet (point clouds)
class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        # shared MLP: the same small network is applied to every point (Conv1d with kernel size 1)
        layers = []
        in_dim = 3    # x, y, z
        for out_dim in POINT_MLP:
            layers = layers + [nn.Conv1d(in_dim, out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        # classifier applied after max pooling
        layers = []
        for out_dim in POINT_HEAD:
            layers = layers + [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(DROPOUT)]
            in_dim = out_dim
        layers = layers + [nn.Linear(in_dim, NUM_CLASSES)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):    # x: (batch, points, 3)
        x = x.transpose(1, 2)              # Conv1d wants (batch, 3, points)
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]         # max over the points: the symmetric function
        return self.head(x)


#Graph CNN (meshes)
class MeshGCN(nn.Module):
    def __init__(self):
        super().__init__()
        from pytorch3d.ops import GraphConv    # imported here so this file also works without PyTorch3D
        self.convs = nn.ModuleList()
        in_dim = 3    # each vertex starts with its x, y, z position
        for out_dim in MESH_HIDDEN:
            self.convs.append(GraphConv(in_dim, out_dim))
            in_dim = out_dim
        self.classifier = nn.Sequential(nn.Linear(in_dim, MESH_FC), nn.ReLU(),
                                        nn.Dropout(DROPOUT), nn.Linear(MESH_FC, NUM_CLASSES))

    def forward(self, meshes):
        x = meshes.verts_packed()      # all vertices of all meshes in the batch, one long list
        edges = meshes.edges_packed()
        for conv in self.convs:
            x = torch.relu(conv(x, edges))
        # average the vertex features of each mesh into one vector per mesh
        mesh_index = meshes.verts_packed_to_mesh_idx()
        sums = torch.zeros(len(meshes), x.shape[1], device=x.device).index_add_(0, mesh_index, x)
        counts = meshes.num_verts_per_mesh().reshape(-1, 1)
        return self.classifier(sums / counts)


#Choose a model by name
def get_model(name):
    if name == "voxel":
        return VoxelCNN()
    if name == "mesh":
        return MeshGCN()
    return PointNet()


if __name__ == "__main__":
    print("voxel:", VoxelCNN()(torch.zeros(4, 1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)).shape)
    print("points:", PointNet()(torch.zeros(4, NUM_POINTS, 3)).shape)
    try:
        from app.data import get_loader
        meshes, labels = next(iter(get_loader("train", "mesh")))
        print("mesh:", MeshGCN()(meshes).shape)
    except ImportError:
        print("mesh: skipped, PyTorch3D is not installed")