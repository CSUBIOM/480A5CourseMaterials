<html>
         <div style="font-size: 20px; width: 1000px;">
              <h1> <left> Welcome to Computational Statistics in Bioengineering! </left> </h1>
              <p><left==========================================left> </p>
              <p> Below is the Github repository holding all the links to Jupyter Notebooks and files needed during the course. </p>
           <h3> Authors: </h3> <pre> Brian Munsky, Luis Aguilera, William Raymond, Dmitri Svetlov, Jack Forman, Joshua Cook, Michael May, Zachary Fox, Alex Popinga, and Eric Ron. </pre>
<p><a href = "mailto: munsky@colostate.edu"> munsky@colostate.edu </a> •  <a href="https://www.engr.colostate.edu/~munsky/">Munsky Group</a> </p>
</div>
</p>                       
</html>

___

- [Course Materials](#course-materials)
- [Python Setup](#python-setup)
  - [Installing Anaconda or Miniconda](#installing-anaconda-or-miniconda)
  - [Creating an Environment](#creating-an-environment)
  - [Using your new environment in VS Code](#using-your-new-environment-in-vs-code)
  - [Learning Python resources](#learning-python-resources)
- [Getting Started with GitHub](#getting-started-with-github)
  - [Helpful GitHub Tutorials:](#helpful-github-tutorials)
- [Licensing](#licensing)


## Course Materials

Below is a summary of the materials for this course.  Please pull to refresh this repository often as these materials may change at any time up to the actual lecture date!  Please note that the six homework assignments cover multiple topics. To complete these assigments follow the provided links, clone the repository and commit and push your work to be graded. See course Canvas page for deadlines.

<left><h3> Module 1 : Getting Started with Basic Scientific Computing in Python. </h3></left>
| Description | Links to Notebook(s) |
| -------- | -------- |
| Hello (Python) world, types, arithmetic operations, iterables, and containers  | [Module 1A ➡️](./Module1-PreliminaryPython/M1A_Intro_to_Python.ipynb) |
| Importing packages, classes/modules, os navigation, file managment basics  | [Module 1B ➡️](./Module1-PreliminaryPython/M1B_Packages_and_File_Management.ipynb) 
| Loops, Ranges, Functions, Lambdas | [Module 1C ➡️](./Module1-PreliminaryPython/M1C_Python_Loops_and_Functions.ipynb) 
| NumPy and linear algebra review | [Module 1D ➡️](./Module1-PreliminaryPython/M1D_NumPy_and_Linear_Algebra.ipynb) 
| Eigenvalue and Eigen Vector Decompositions | [Module 1E ➡️](./Module1-PreliminaryPython/M1E_Eigen_Values_Vectors_Decompositions_and_Transformations.ipynb) 
| Visualizing biological data using Matplotlib | [Module 1F ➡️](./Module1-PreliminaryPython/M1F_Matplotlib.ipynb) 
___

<left><h3> Module 2 :  Introduction to Basic Bioinformatics.  </h3></left>
 | Description | Notebook(s)  |
 | -------- | -------- |
 | Finding, Downloading, and Annotating Biological Sequences in Python | [Module 2A ➡️](./Module2-Bioinformatics/M2A_Getting_and_Annotating_Genomic_Sequences.ipynb) |
 | Aligning Pairs and Sets of Biological Sequences in Python | [Module 2B ➡️](./Module2-Bioinformatics/M2B_Aligning_Sequences.ipynb) |
 | Performing BLAST Searches to Discover Related Sequences | [Module 2C ➡️]([./Module2-Bioinformatics/M2C_Finding_Related_Sequences.ipynb) |
 | Constructing Phylogenetic Trees to Represent Biological Histories | [Module 2D ➡️](./Module2-Bioinformatics/M2D_Constructing_Phylogenetic_Histories.ipynb) |

___

<left><h3> Module 3 :  Multivariable Statistics and Machine Learning for Biological Data.  </h3></left>
| Description  | Notebook(s)  | 
| -------- | -------- |
 | Sampling from Distributions, Expected Values and Summary Statistics | [Module 3A ➡️](./Module3-Statistics/M3A_Distributions_Sampling_Expectations_and_SummaryStatistics.ipynb) |
 | Computing Likelihood Functions and Estimating Models from Data| [Module 3B ➡️]([./Module3-Statistics/M3B_LikelihoodFunctions_Estimation.ipynb) |
 | Performing Hypothesis Testing to Determine Significance and P-values | [Module 3C ➡️](./Module3-Statistics/M3C_HypothesisTesting_Significance_Pvalues.ipynb) |
 | Introduction to Machine Learning and Regression Analyses| [Module 3D ➡️](./Module3-Statistics/M3D_Intro_to_ML_and_Basics_of_Regression.ipynb) |
 | Using Machine Learning for Classification Analysis| [Module 3E ➡️](./Module3-Statistics/M3E_Basics_of_Classification.ipynb) |
___


<left><h3> Module 4 : Optical Microscopy Experiments and Image Processing. </h3></left>
| Description | Notebook(s)  |
| -------- | -------- |
| Basic Steps in Image Loading and Processing | [Module 4A ➡️](./Module4-ImageProcessing/M4A_Basic_Image_Processing.ipynb)   |
| Using Machine Learning to Process Handwritten Digits | [Module 4B ➡️](./Module4-ImageProcessing/M4B_Processing_Handwrtten_Digits.ipynb) |
| Segmenting Images to Extract and Quantify Cellular Features  | [Module 4C ➡️](./Module4-ImageProcessing/M4C_Segmenting_Images.ipynb) |
| Tracking Biological Particles to Explore Sub-Cellular Motion | [Module 4D ➡️](./ImageProcessing/M4D_Tracking_Particles.ipynb) 
___

<left><h3> Module 5 :  Computational Modeling and Stochastic Simulation of Biological Processes. </h3></left>
| Description | Notebook(s)  |
| --------| -------- |
| Stoichiometries, Propensity Functions, and ODE Models | [Module 5A ➡️](./Module5-ModelingBiochemicalReactions/M5A_Stoichiometries_Propensities_and_ODE_Models.ipynb)|
| Generating Stochastic Simulations of Cellular Trajectories| [Module 5B ➡️](./Module5-ModelingBiochemicalReactions/M5B_Stochastic_Simulation_Algorithm.ipynb) |
| Chemical Master Equation and the Finite State Projection| [Module 5C ➡️](./Module5-ModelingBiochemicalReactions/M5C_Chemical_Master_Equation.ipynb)|
___

<left><h3> Module 6 :  Bayesian Model Inference and Markov Chain Monte Carlo (MCMC) Methods. </h3></left>
| Description | Notebook(s)  |
| --------| -------- |
| Formulating Bayes's Theorem to Analyze Data in Python | [Module 6A ➡️](./Module6-MCMC/M6A_Bayesian_Thinking.ipynb)|  
| Using Markov Chain Monte Carlo to Solve Complex Problems | [Module 6B ➡️](./Module6-MCMC/M6B_Markov_Chain_Monte_Carlo.ipynb)|  
___

## Python Setup 

### Installing Anaconda or Miniconda

First we need an environment manager! Using an environment manager is good practice when doing coding projects as it lets you setup seperate coding environments and kernels for different tasks. 

Download and install either anaconda or miniconda from:  https://www.anaconda.com/download. Anaconda includes more packages and features that come with it but is around 5 gigabytes to install. Miniconda installs just the bare necessities and is 400 mb. For our purposes either works.

### Creating an Environment

Once you have conda installed, we need to create a new environment for the class. 

* Restart your computer so conda is available.
* Open up a new Terminal (Linux/Mac) or Anaconda Prompt / Command Prompt (Windows) and type the following commands (Note - in the screen shots provided below, we used the environment name "uqbio2024_py310," but this can be changed to suit your personal preferences.)
   1. ```conda activate```  - after activating you will see (base) in front of your prompt, this is the current conda environment you are in. 

   2. ```conda create --name "compstatbioeng" python=3.10``` 
   ![](./uqbio2024_files/step1.png)

   3. ```conda activate compstatbioeng```
   ![](./uqbio2024_files/step3.png) you should see your current environment switch to (compstatbioeng)

   4. ```conda install numpy pandas scipy jupyter ipython matplotlib pillow scikit-image seaborn tifffile beautifulsoup4 scikit-learn conda-forge::trackpy```
   ![](./uqbio2024_files/step4.png)
      * Note this command will take a while and will ask for your confirmation!

   5. ```pip install biopython```
   ![](./uqbio2024_files/step5.png)

   7. Last thing to install is ```cellpose``` which is a little involved and requires ```pytorch```. 
      * For **GPU** acceleration, you need an nvidia graphics card and a CUDAtoolkit to check these go to the following website: https://pytorch.org/get-started/locally/ and run the command given there for your system. PLEASE REMOVE ```torchvision``` and ```torchaudio``` from the command! This will save you time and space as we are not using those packages. After CUDA and ```torch``` are installed, run the following command to install cellpose: ```pip install git+https://www.github.com/mouseland/cellpose.git```
      * For CPU if you do not have an nvidia graphics card, just run the following: ```pip install git+https://www.github.com/mouseland/cellpose.git```
   7. Finally check your environment is working correctly by running the ```check_environment.py``` file in this repository. Navigate to the directory where your local repository is and run: ```python check_environment.py```. If there are no error messages then you are good to go!

### Using your new environment in VS Code

Now that we have a brand new environment, we can use an IDE (integrated developing environment) to use this Python environment for Python coding. For this course we recommend VS Code: [Visual Studio](https://code.visualstudio.com/)

1. Open VS Code
![](./uqbio2024_files/vscode1.png)
   * Optional: Make a new workspace under File > Workspace to your CourseMaterials folder

2. Enter the search bar and type ">" or use "Ctrl + Shift + P" for activating the command palette
![](./uqbio2024_files/vscode1a.png)

3. type Python: Select interpreter and select the compstatbioeng kernel
![](./uqbio2024_files/vscode2.png)

5. At the top, click Terminal and then New terminal or use "Ctrl + Shift + `"
![](./uqbio2024_files/vscode3.png)

6. Check that the terminal that popped up is using our correct environment by making sure it says ```(compstatbioeng)``` in front of your path name in the terminal or command prompt.
![](./uqbio2024_files/vscode4.png)

Feel free to check out the preliminary python notebooks under CourseMaterials/Module1-PreliminaryPython/. To run a notebook file in VS Code, simply open the .ipynb file in VS code by navigating to it in your workspace or using File > Open File.

### Learning Python resources

[Python 3.10 documentation's tutorial](https://docs.python.org/3.10/tutorial/index.html)

## Getting Started with GitHub

We are going to use and share a lot of codes during the course, and it can be difficult to keep track of who is working on what.  To help us with this common concern, we are going to use GitHub for version control. 

This is also how course homework will be assigned, submitted, and graded throughout the course. So, if you want fast feedback, it is very important for you to learn how to use GitHub!

Please follow these steps to get started using GitHub (and to complete your first homework assignment):

**Step 1** Visit [https://github.com/](https://github.com/) and sign up for a free account. Keep track of your user name and password, and you may need to set up two-factor identification. Follow instructions to do so.

**Step 2** If you are not an expert on GitHub already, I recommend installing and using the GitHub Desktop Client for performing necessary GitHub tasks. Download this at: [https://desktop.github.com](https://desktop.github.com).

**Step 3** Once you have installed GitHub Desktop, launch it, and it will ask you to sign in as a GitHub user (use your account from Step 1).

**Step 4** You will reach a page titled "Let's get started". Click the button "clone a repository" on the second from the top on the left. Type your repository name ("CSUBIOM/CompStatBioeng")
Click the button "Clone" at the bottom.

If all went well, you should now have all of the course codes and data on your computer, and you should be able to follow along with our class exercises and be able to complete the first part of the first homework assignment.

## Homework Assignments

**If you are taking this course for credit at CSU, please see instructions on CANVAS for how to proceed and get access to your homework assignments, discussion boards, and group challenges. SKIP THE REST OF THIS SECTION - THESE ASSIGNMENTS ARE ONLY FOR NON-CSU PARTICPANTS AND WILL NOT BE GRADED FOR CSU CREDIT.**

Once you have a GitHub account, you can sign up for the first Homework Assignment - Introduction to GitHub and Python. To get access, you must sign up for the UQ-Bio At Home Course at [https://forms.gle/6VSxaoSCyqmPvg9q8](https://forms.gle/6VSxaoSCyqmPvg9q8).  

Upon completing the application, you will be given a link to the first homework assignment. That link will take you to a signup sheet for the class. Click "skip to the next step". Refresh a few times until your repository is created and then follow the links.

Once the repository is ready, a personalized link will appear.  Follow this link to your own personal repository to see your first assignment.

### Helpful GitHub Tutorials:
Tutorial on getting started in GitHub Desktop: [GitHub Desktop Tutorial](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop)

Suggestions for editors to use for modifying your README file:
* You can edit the readme directly in VSCode.
* Windows users should be able to edit the README in Notepad++, which should be automatically installed.
* Mac users should be able to edit it in TextEdit, which should come pre-installed.
* You can also edit readme directly on the GitHub page, which is what I usually do.

[Slides discussing Git and GitHub philosophy and best practices](uqbio2024_files/GitAndGitHubPresentation_2024UQ-BioSummerSchool.pdf)

Cheatsheet with GitHub Commands: [github-git-cheatsheet.pdf](https://education.github.com/git-cheat-sheet-education.pdf)

**Thank you for reading this far!** As a reward, please click the image below for a special video of Q-Bio poetry from Brian:

[![Poetry Video from Brian](https://img.youtube.com/vi/MrPRz1_4euA/0.jpg)](https://www.youtube.com/watch?v=MrPRz1_4euA)

## Licensing

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)



