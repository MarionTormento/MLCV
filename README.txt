Random Forest and Visual Codebook with Caltech101
by Edward McLaughlin and Marion Tormento on 14/03/2018.

This toolbox is made for demonstrating randomised decision forest (RF) on several 
toy data sets and Caltech101 image categorisation data set.

---------------------------------------------------------------------------
This code is inspired by Mang Shao and Tae-Kyun Kim toolbox on Random Forests:
---------------------------------------------------------------------------
The folder refers to
    Images          - images computed for the report
    Question_1_2    - code to be computed for question 1 and 2
    Question_3      - code to be computed for question 3
    RF2018          - code provided by Mang Shao and Tae-Kyun Kim / Must be run to obtain training and testing data for Q3
    varycolor       - library for colors in plots

The main scripts to run (from each question directory) is:
    main.m          - run script
 
---------------------------------------------------------------------------   
Some important functions:
    
question functions:

    trainForest.m       - Grow random forest, each decision tree has the following structure:

           Base               Each tree [nodes{1,k}] stores for each node [nodes{1,k}{layer nb,position in the layer}]:
          n=1                   .x1            - 'X', 'Y' if axis aligned or gradient if linear split or 'Leaf'
           / \                  .x2            - value for each axis aligned, or intercept if linear split
          2   3                 .Gain          - Information gain
         / \ / \                .dim           - only when runing Q3, dimension of the splitting plan
        4  5 6  7               
       ...........            Each leaf node [leaves(m)]
                                leaves(m,1)    - Tree position of the leaf node
                                leaves(m,2)    - Layer position of the leaf node
                                leaves(m,3)    - Position in the layer of the leaf node
                                leaves(m,4:end)- probabilistic class distribution (1:3/1:10) of data in this leaf node 
                              
    optimalNodeSplit.m   - Chose between the axis split node and the linear split node depending on the information gain
    axisNodeSplit.m      - Split node for an axis split weak learner
    linearNodeSplit.m    - Split node for a linear split weak learner

    testForest.m         - Run random forest on testing data to obtain the predicted class

    buildCodebook.m      - build Codebook for question 3-3 using the random forest
    testCodebook.m       - test the RF built with the builtCodebook to obtain the training and test data for Q3-3
 
internal functions:

    getData.m           - Generate training and testing data for question 1, 2, 3-1, 3-2
   
    getDataRF.m         - Generate training and testing data with a RF codebook for question 3-3
    
external functions and libraries:
    
    VLFeat    - A large open source library implements popular computer vision algorithms. BSD License.
                http://www.vlfeat.org/index.html
    
    subaxis.m - Modified 'subplot' function. No BSD License
  parseArgs.s   Written by Aslak Grinsted
                http://www.mathworks.co.uk/matlabcentral/fileexchange/3696-subaxis-subplot

    suptitle.m- Create a "master title" at the top of a figure with several subplots
                Written by Drea Thomas

    Caltech_101 image categorisation dataset
                    L. Fei-Fei, R. Fergus and P. Perona. Learning generative visual models
                    from few training examples: an incremental Bayesian approach tested on
                    101 object categories. IEEE. CVPR 2004, Workshop on Generative-Model
                    Based Vision. 2004
                http://www.vision.caltech.edu/Image_Datasets/Caltech101/
---------------------------------------------------------------------------
