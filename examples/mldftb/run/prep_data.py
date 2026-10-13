import ase.io
import numpy as np
import numpy.random

np.random.seed(0)

train_ = ase.io.read("/Users/will/PhDwork/scf/metal_water_interfaces/all_download_csd3/metal_water/data/train_test/keys_combined_slabs_and_molecules_train.xyz", index=':')
train_ = [item for item in train_ if (not (np.any(item.pbc)) and not "C" in item.get_chemical_symbols())]
test_ = ase.io.read("/Users/will/PhDwork/scf/metal_water_interfaces/all_download_csd3/metal_water/data/train_test/keys_combined_slabs_and_molecules_test.xyz", index=':')
test_ = [item for item in test_ if (not (np.any(item.pbc)) and not "C" in item.get_chemical_symbols())]

#train = np.random.choice(train_, 1000, replace=False)
#test = np.random.choice(test_, 100, replace=False)

train=train_
test=test_

def prep_data(ats_list):
    for item in ats_list:
        item.info["elec_temp"] = 0.025
        charge = item.info['total_charge']
        electrons = round(item.get_chemical_symbols().count("H") + 2*item.get_chemical_symbols().count("O"))
        item.info["N_alpha"] = np.sum(electrons)//2
        item.info["N_beta"] = np.sum(electrons)//2

prep_data(train)
prep_data(test)

ase.io.write("train.xyz", train)
ase.io.write("test.xyz", test)