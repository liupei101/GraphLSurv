# S01: Pathology Images Preprocessing

Subtasks in this stage are listed as follows:
- Dataset explore
- Images tissue segmentation and tiling
- Patch Feature extraction
- Overview of Statistics 

Three directories presented in current path are:
- **tools**: We fork [CLAM](https://github.com/mahmoodlab/CLAM) and make it suitable for our task. It implements all preprocessing procedures, such as tissue segmentation, WSI patching, patch feature extraction. Moreover, it offers command-line interfaces for users to conveniently process pathological images.
- **scripts**: It mainly provides essential command-line to process WSIs.
- **notebooks**: Notebooks explain each preprocessing part and show the results of them.
