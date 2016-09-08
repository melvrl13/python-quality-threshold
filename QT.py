import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg') #For use on headless machine
import matplotlib.pyplot as plt
plt.style.use('bmh')
import tempfile
import argparse

# Initialize parser. The default help has poor labeling. See http://bugs.python.org/issue9694
parser = argparse.ArgumentParser(description = 'Runs a vectorized version of QT clustering', add_help=False) 

# List all possible user input
inputs=parser.add_argument_group('Input arguments')
inputs.add_argument('-h', '--help', action='help')
inputs.add_argument('-top', action='store', dest='structure',help='Structure file corresponding to trajectory',type=str,required=True)
inputs.add_argument('-traj', action='store', dest='trajectory',help='Trajectory',type=str,required=True)
inputs.add_argument('-sel', action='store', dest='sel', help='Atom selection',type=str,default='name CA')
inputs.add_argument('-min', action='store', dest='minimum_membership', help='Minimum number of frames in a cluster',type=int,default=2)
inputs.add_argument('-cutoff', action='store', dest='cutoff', help='maximum cluster radius',type=float,required=True)
inputs.add_argument('-o', action='store', dest='out_name',help='Output directory',type=str,required=True)

# Parse into useful form
UserInput=parser.parse_args()

topology = UserInput.structure
trajectory = UserInput.trajectory
t = md.load(trajectory,top=topology)
n_frames = t.n_frames
sel = t.topology.select(UserInput.sel)
t = t.atom_slice(sel)

tempfile = tempfile.NamedTemporaryFile()
distances = np.memmap(tempfile.name, dtype=float, shape=(n_frames,n_frames))
#distances = np.empty((n_frames, n_frames), dtype=float)
t.center_coordinates()
for i in range(n_frames):
    distances[i] = md.rmsd(target=t, reference=t, frame=i, precentered=True)

t = None
cutoff_mask = distances <= UserInput.cutoff
distances = None
centers = []
cluster = 0
labels = np.empty(n_frames)
labels.fill(np.NAN)

while cutoff_mask.any():
    membership = cutoff_mask.sum(axis=1)
    center = np.argmax(membership)
    members = np.where(cutoff_mask[center,:]==True)
    if max(membership) <= UserInput.minimum_membership:
        labels[np.where(np.isnan(labels))] = -1
        break
    labels[members] = cluster
    centers.append(center)
    cutoff_mask[members,:] = False
    cutoff_mask[:,members] = False
    cluster = cluster + 1


# Save results
#Text
np.savetxt(UserInput.out_name + '/QT_labels.txt', labels, fmt='%i')
np.savetxt(UserInput.out_name + '/QT_centers.txt', centers, fmt='%i')

#Figures
plt.figure()
plt.scatter(np.arange(n_frames), labels, marker = '+')
plt.xlabel('Frame')
plt.ylabel('Cluster')
plt.title('QT')
plt.savefig(UserInput.out_name + '/QT.png')
plt.close()
