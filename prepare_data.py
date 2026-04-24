from ulab_data_edited import load_data  
import numpy as np                                                                                      
train_dataset, _ = load_data("training 3")                                                              
test_dataset, _ = load_data("testing 3", [train_dataset.pmeans, train_dataset.pstds])                   
np.savez("test_data.npz",                                                                               
    pmeans=train_dataset.pmeans,  # IMPORTANT: training stats, not test                                 
    pstds=train_dataset.pstds,
    file_paths=np.array(test_dataset.file_paths),
    curve_list=np.array(test_dataset.curve_list, dtype=object),
    first_times=np.array(test_dataset.first_times),
    i_bl_guesses=np.array(test_dataset.i_bl_guesses),
)