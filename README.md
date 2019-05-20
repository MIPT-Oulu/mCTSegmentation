# Deep Learning-based 3D Segmentation of Calcified Cartilage Interface in Human Osteochondral Samples Imaged with Micro-Computed Tomography

*Aleksei Tiulpin, Tuomas Frondelius, Heikki J. Nieminen, Petri Lehenkari, Simo Saarakkala*

## Installation
```
conda create -f pta_segmentation.yml
```

## Training
The script below will download the data, execute the experiments (it will take several days on 3xGTX1080Ti) 
and eventually generate the result pictures presented below. 

```
sh run_experiments.sh
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
