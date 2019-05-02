# Deep Learning-based 3D Segmentation of Calcified Cartilage Interface in Human Osteochondral Samples Imaged with Micro-Computed Tomography

*Aleksei Tiulpin, Tuomas Frondelius, Heikki J. Nieminen, Petri Lehenkari, Simo Saarakkala*

## Data

The data acompanying this paper can be downloaded as follows: `sh download_data.sh`. 
By default, it will create a folder in `DATA_DIR` that can be changed.

## Installation
```
pip install git+https://github.com/MIPT-Oulu/KVS.git
pip install -e .
```

## Results

<table style="width:100%">
  <tr>
    <td><img src="pics/IoU.png" width="300" /> </td>
    <td><img src="pics/Dice.png" width="300"/></td>
    <td><img src="pics/VS.png" width="300"/></td>  
  </tr>
  <tr>
    <td align="center">IoU</td>
    <td align="center">Dice</td>
    <td align="center">Volumetric Similarity</td>
  </tr>
</table>
