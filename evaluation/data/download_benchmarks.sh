#!/bin/bash

set -e

dir_prefix=$1

if [ ! -d $dir_prefix ] && echo "Directory ${dir_prefix} DOES NOT exists."; then
    mkdir $dir_prefix
fi

lucchi_dir="${dir_prefix}/lucchi_pp/"
if [ ! -d $lucchi_dir ] && echo "Directory ${lucchi_dir} DOES NOT exists."; then
    echo "Downloading Lucchi++ ..."
    mkdir $lucchi_dir
    wget -q http://www.casser.io/files/lucchi_pp.zip -P $lucchi_dir
    unzip -qq $lucchi_dir/lucchi_pp.zip -d $lucchi_dir
    rm $lucchi_dir/lucchi_pp.zip
    mv $lucchi_dir/Lucchi++/* $lucchi_dir
    rm -r $lucchi_dir/Lucchi++/
    mkdir $lucchi_dir/train $lucchi_dir/test
    mv $lucchi_dir/Train_In $lucchi_dir/train/images
    mv $lucchi_dir/Train_Out $lucchi_dir/train/masks
    mv $lucchi_dir/Test_In $lucchi_dir/test/images
    mv $lucchi_dir/Test_Out $lucchi_dir/test/masks
else
    echo "Lucchi++ already downloaded, skipping"
fi

kasthuri_dir="${dir_prefix}/kasthuri_pp/"
if [ ! -d $kasthuri_dir ] && echo "Directory ${kasthuri_dir} DOES NOT exists."; then
    echo "Downloading Kasthuri++ ..."
    mkdir $kasthuri_dir
    wget -q http://www.casser.io/files/kasthuri_pp.zip -P $kasthuri_dir
    unzip -qq $kasthuri_dir/kasthuri_pp.zip -d $kasthuri_dir
    rm $kasthuri_dir/kasthuri_pp.zip
    mv $kasthuri_dir/Kasthuri++/* $kasthuri_dir
    rm -r $kasthuri_dir/Kasthuri++/
    mkdir $kasthuri_dir/train $kasthuri_dir/test
    mv $kasthuri_dir/Train_In $kasthuri_dir/train/images
    mv $kasthuri_dir/Train_Out $kasthuri_dir/train/masks
    mv $kasthuri_dir/Test_In $kasthuri_dir/test/images
    mv $kasthuri_dir/Test_Out $kasthuri_dir/test/masks
else
    echo "Kasthuri++ already downloaded, skipping"
fi

perez_dir="${dir_prefix}/perez/"
if [ ! -d $perez_dir ] && echo "Directory ${perez_dir} DOES NOT exists."; then
    echo "Downloading Perez ..."
    mkdir $perez_dir
    wget -q https://www.sci.utah.edu/releases/chm_v2.1.367/chm-supplemental_data.zip -P $perez_dir
    unzip -qq $perez_dir/chm-supplemental_data.zip -d $perez_dir
    rm $perez_dir/chm-supplemental_data.zip
    mv $perez_dir/supplemental_data/* $perez_dir
    rm -r $perez_dir/supplemental_data/
    
    #directories for organelles
    mkdir $perez_dir/mito $perez_dir/lyso $perez_dir/nuclei $perez_dir/nucleoli
    
    #make train and test dirs in each
    mkdir $perez_dir/mito/train $perez_dir/mito/test
    mkdir $perez_dir/lyso/train $perez_dir/lyso/test
    mkdir $perez_dir/nuclei/train $perez_dir/nuclei/test
    mkdir $perez_dir/nucleoli/train $perez_dir/nucleoli/test

    mv $perez_dir/training_images/mitochondria $perez_dir/mito/train/images
    mv $perez_dir/training_labels/mitochondria $perez_dir/mito/train/masks
    mv $perez_dir/test_images/mitochondria $perez_dir/mito/test/images
    mv $perez_dir/ground_truth/mitochondria $perez_dir/mito/test/masks
    
    mv $perez_dir/training_images/lysosomes $perez_dir/lyso/train/images
    mv $perez_dir/training_labels/lysosomes $perez_dir/lyso/train/masks
    mv $perez_dir/test_images/lysosomes $perez_dir/lyso/test/images
    mv $perez_dir/ground_truth/lysosomes $perez_dir/lyso/test/masks
    
    mv $perez_dir/training_images/nuclei $perez_dir/nuclei/train/images
    mv $perez_dir/training_labels/nuclei $perez_dir/nuclei/train/masks
    mv $perez_dir/test_images/nuclei $perez_dir/nuclei/test/images
    mv $perez_dir/ground_truth/nuclei $perez_dir/nuclei/test/masks
    
    mv $perez_dir/training_images/nucleoli $perez_dir/nucleoli/train/images
    mv $perez_dir/training_labels/nucleoli $perez_dir/nucleoli/train/masks
    mv $perez_dir/test_images/nucleoli $perez_dir/nucleoli/test/images
    mv $perez_dir/ground_truth/nucleoli $perez_dir/nucleoli/test/masks
    
    rm -r $perez_dir/training_images $perez_dir/training_labels $perez_dir/test_images $perez_dir/ground_truth

else
    echo "Perez already downloaded, skipping"
fi

guay_dir="${dir_prefix}/guay/"
if [ ! -d $guay_dir ] && echo "Directory ${guay_dir} DOES NOT exists."; then
    echo "Downloading Guay ..."
    mkdir $guay_dir
    wget -q https://www.dropbox.com/s/68yclbraqq1diza/platelet_data_1219.zip -P $guay_dir
    unzip -qq $guay_dir/platelet_data_1219.zip -d $guay_dir
    
    rm $guay_dir/platelet_data_1219.zip
    mv $guay_dir/platelet_data/ $guay_dir/3d/
    mkdir $guay_dir/2d
    
    mkdir $guay_dir/3d/train $guay_dir/3d/valid $guay_dir/3d/test
    mkdir $guay_dir/3d/train/images $guay_dir/3d/train/masks
    mkdir $guay_dir/3d/valid/images $guay_dir/3d/valid/masks
    mkdir $guay_dir/3d/test/images $guay_dir/3d/test/masks
    
    mv $guay_dir/3d/train-images.tif $guay_dir/3d/train/images
    mv $guay_dir/3d/train-labels.tif $guay_dir/3d/train/masks
    mv $guay_dir/3d/eval-images.tif $guay_dir/3d/valid/images
    mv $guay_dir/3d/eval-labels.tif $guay_dir/3d/valid/masks
    mv $guay_dir/3d/test-images.tif $guay_dir/3d/test/images
    mv $guay_dir/3d/test-labels.tif $guay_dir/3d/test/masks
    rm $guay_dir/3d/train-error-weights.tif
    
else
    echo "Guay already downloaded, skipping"
fi

urocell_dir="${dir_prefix}/urocell/"
if [ ! -d $urocell_dir ] && echo "Directory ${urocell_dir} DOES NOT exists."; then
    echo "Downloading UroCell ..."
    mkdir $urocell_dir
    mkdir $urocell_dir/2d $urocell_dir/3d
    mkdir $urocell_dir/3d/train $urocell_dir/3d/test
    mkdir $urocell_dir/3d/train/images $urocell_dir/3d/train/masks $urocell_dir/3d/train/mito $urocell_dir/3d/train/lyso
    mkdir $urocell_dir/3d/test/images $urocell_dir/3d/test/masks $urocell_dir/3d/test/mito $urocell_dir/3d/test/lyso
    
    
    #training sets
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-1-0-3.nii.gz -P $urocell_dir/3d/train/images
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-1-0-3.nii.gz -P $urocell_dir/3d/train/lyso
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-1-0-3.nii.gz -P $urocell_dir/3d/train/mito
    
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-3-2-1.nii.gz -P $urocell_dir/3d/train/images
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-3-2-1.nii.gz -P $urocell_dir/3d/train/lyso
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-3-2-1.nii.gz -P $urocell_dir/3d/train/mito
        
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-3-3-0.nii.gz -P $urocell_dir/3d/train/images
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-3-3-0.nii.gz -P $urocell_dir/3d/train/lyso
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-3-3-0.nii.gz -P $urocell_dir/3d/train/mito
    
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-4-3-0.nii.gz -P $urocell_dir/3d/train/images
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-4-3-0.nii.gz -P $urocell_dir/3d/train/lyso
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-4-3-0.nii.gz -P $urocell_dir/3d/train/mito
    
    #test set
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/data/fib1-0-0-0.nii.gz -P $urocell_dir/3d/test/images
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/lyso/fib1-0-0-0.nii.gz -P $urocell_dir/3d/test/lyso
    wget -q https://github.com/MancaZerovnikMekuc/UroCell/raw/master/mito/fib1-0-0-0.nii.gz -P $urocell_dir/3d/test/mito
else
    echo "UroCell already downloaded, skipping"
fi

cremi_dir="${dir_prefix}/cremi/"
if [ ! -d $cremi_dir ] && echo "Directory ${cremi_dir} DOES NOT exists."; then
    echo "Downloading CREMI ..."
    mkdir $cremi_dir
    mkdir $cremi_dir/2d $cremi_dir/3d
    mkdir $cremi_dir/3d/train $cremi_dir/3d/test
    mkdir $cremi_dir/3d/train/images $cremi_dir/3d/train/masks
    mkdir $cremi_dir/3d/test/images $cremi_dir/3d/test/masks
    
    wget -q https://cremi.org/static/data/sample_A_20160501.hdf -P $cremi_dir/3d/train
    wget -q https://cremi.org/static/data/sample_B_20160501.hdf -P $cremi_dir/3d/train
    wget -q https://cremi.org/static/data/sample_C_20160501.hdf -P $cremi_dir/3d/test
else
    echo "CREMI already downloaded, skipping"
fi