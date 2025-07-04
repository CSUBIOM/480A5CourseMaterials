#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 08:04:38 2024

@author: munsky
"""

import numpy as np

def chooseMember(students, probs):
    if np.sum(probs)==0:
        print('All probabilities are zero - Nothing to choose!')
        return [], []
        
    # Ensure that probabilities sum to 1
    probs = probs / np.sum(probs)
    
    # Randomly select an entry based on the probability mass function
    student = np.random.choice(students, p=probs)
    
    # Get the index of the chosen entry
    index = students.index(student)
    
    return student, index

def spring2025Students():
    return ['Becca B','Ella B','Lauren B','James B','Chayanee C','Deven D','Jackson E','Delaney E','Christian F','Lillian G','Andy G',
            'Emma G','Nick G','Joshua H','Paycen H','Kyle H','Emma H','Elijah J','Gabriela J','Addison L',
            'Matthew L','Hassan M','Bella P','Chris RT','Mariana S','Dylan S','Jacob S','Hayley S','Kayla T',
            'Vivia VDM','Fig V','AbbyMae W','Alvina Y','Polina Z']

def geneRegulationTopics():
    return ['Signal Transduction and Cellular Signaling - Receptor-Mediated Signaling: Activation of gene expression in response to extracellular signals.',
'Signal Transduction and Cellular Signaling - Intracellular Signaling Cascades: Series of molecular events triggered by signaling molecules.',
'Signal Transduction and Cellular Signaling - Second Messenger Systems: Involvement of molecules like cAMP, cGMP, and calcium ions.',
'Signal Transduction and Cellular Signaling - Phosphorylation/Dephosphorylation: Post-translational modification of proteins to activate or deactivate signaling pathways.',
'Signal Transduction and Cellular Signaling - G-protein Coupled Receptors (GPCRs): Activation of gene expression through GPCR-mediated pathways.',
'Chromatin Remodeling and Epigenetic Regulation - Histone Modification: Acetylation, methylation, phosphorylation of histone proteins.',
'Chromatin Remodeling and Epigenetic Regulation - DNA Methylation: Addition of methyl groups to DNA, influencing gene expression.',
'Chromatin Remodeling and Epigenetic Regulation - Chromatin Remodeling Complexes: Alteration of chromatin structure to allow or restrict transcription.',
'Chromatin Remodeling and Epigenetic Regulation - Non-Coding RNAs (ncRNAs): Involvement of microRNAs (miRNAs) and long non-coding RNAs (lncRNAs).',
'Promoter Activation and Transcription Initiation - Promoter Binding Proteins: Transcription factors binding to gene promoters.',
'Promoter Activation and Transcription Initiation - Enhancers and Silencers: DNA elements influencing transcription positively (enhancers) or negatively (silencers).',
'Promoter Activation and Transcription Initiation - Transcriptional Coactivators/Co-repressors: Proteins that enhance or inhibit transcriptional activity.',
'Promoter Activation and Transcription Initiation - RNA Polymerase Recruitment: Binding of RNA polymerase to gene promoters.',
'Promoter Activation and Transcription Initiation - Mediator Complex: Bridges between transcription factors and RNA polymerase.',
'Transcription Elongation and Termination - Elongation Factors: Proteins facilitating RNA polymerase movement along the DNA.',
'Transcription Elongation and Termination - Pausing and Anti-Pausing Factors: Regulation of RNA polymerase pausing and elongation.',
'Transcription Elongation and Termination - Terminator Sequences: Signals marking the end of transcription.',
'RNA Processing and Modification - RNA Splicing: Removal of introns and joining of exons.',
'RNA Processing and Modification - Polyadenylation: Addition of poly-A tail to mRNA.',
'RNA Processing and Modification - RNA Editing: Alteration of nucleotide sequence in RNA transcripts.',
'mRNA Transport and Localization - Nuclear Export: Transport of mature mRNA from the nucleus to the cytoplasm.',
'mRNA Transport and Localization - mRNA Localization Signals: Sequences directing mRNA to specific cellular locations.',
'Translation Initiation and Elongation - Ribosome Binding Sites: Recognition of mRNA by ribosomes.',
'Translation Initiation and Elongation - Initiator tRNA: tRNA carrying the first amino acid in translation.',
'Translation Initiation and Elongation - eIFs (Eukaryotic Initiation Factors): Proteins facilitating translation initiation.',   
'Translation Initiation and Elongation - Termination Factors: Proteins involved in translation termination.',
'Post-Translational Modification - Protein Phosphorylation: Addition of phosphate groups to proteins.',
'Post-Translational Modification - Ubiquitination: Addition of ubiquitin molecules for protein degradation.',
'Post-Translational Modification - Acetylation/Deacetylation: Addition or removal of acetyl groups to regulate protein function.', 
'Post-Translational Modification - Glycosylation: Addition of sugar moieties to proteins.',
'Protein Folding and Stability - Chaperone Proteins: Assist in proper protein folding.',    
'Protein Folding and Stability - Proteasomal Degradation: Targeted degradation of proteins by the proteasome.',
'Cellular Localization - Nuclear Localization Signals (NLS): Sequences directing proteins to the nucleus.',    
'Cellular Localization - Mitochondrial Targeting Signals: Sequences directing proteins to mitochondria.',    
'Cellular Localization - Endoplasmic Reticulum (ER) Targeting Signals: Sequences directing proteins to the ER.',
'Feedback Regulation and Feedback Inhibition - Negative Feedback Loops: Mechanisms to dampen excessive gene expression.',    
'Feedback Regulation and Feedback Inhibition - miRNA-mediated Regulation: Post-transcriptional regulation by microRNAs.',
'Environmental and Metabolic Regulation - Nutrient Availability: Availability of nutrients influencing gene expression.',
'Environmental and Metabolic Regulation - Oxygen and Redox Regulation: Cellular oxygen levels affecting gene expression.', 
'Environmental and Metabolic Regulation - Metabolite Sensing: Cellular metabolite levels impacting gene regulation.',
'Stress Response - Heat Shock Proteins (HSPs): Induction in response to cellular stress.',
'Stress Response - DNA Damage Response: Activation of repair mechanisms in response to DNA damage.',
'Hormonal Regulation - Steroid Hormones: Influence gene expression through nuclear receptors.',    
'Hormonal Regulation - Peptide Hormones: Signaling cascades activated by peptide hormones.',
'Epistasis and Genetic Interaction - Genetic Interactions: Interaction between genes influencing expression patterns.',    
'Epistasis and Genetic Interaction - Epistasis: Gene interaction where the effect of one gene depends on the presence of another.',
'Non-Genetic Inheritance - Transgenerational Epigenetic Inheritance: Inheritance of epigenetic marks across generations.',
'Cell Cycle Regulation - Cyclins and Cyclin-Dependent Kinases (CDKs): Regulate gene expression during the cell cycle.',
'Cellular Differentiation - Master Regulator Genes: Genes controlling cell fate determination and differentiation.',   
'Cell Signaling Gradients - Gradients of signaling molecules influencing gene expression patterns.']


def pythonModules():
    return ['pysam',
            'Chimera',
            'MAFFT',
            'AlphaFold 2',
            'Phenix',
            'GATK',
            'metaspades',
            'SPAdes',
            'HTSeq',
            'htseqcount',
            'QIIME 2',
            'MACS',
            'MODELLER',
            'BUSCO',
            'BEDTools',
            'PICRUSt',
            'Avogadro',
            'CheckM',
            'QUAST',
            'HISAT2',
            'kallisto',
            'Unicycler',
            'GENCODE',
            'Expasy',
            'MultiQC',
            'BioBERT',
            'ScanPy',
            'ColabFold',
            'Open Babel',
            'PICRUSt2',
            'LDSC',
            'scRNA-Seq',
            'GTDB-Tk',
            'Flye',
            'UNet++',
            'SBML',
            'HyPhy',
            'Scoary',
            'NCBI Mass Sequence Downloader',
            'CompreheNGSive',
            'PAIRPred',
            'Omics Pipe',
            'GROMACS',
            'OrthoFinder',
            'DeepLabCut',
            'MiRPursuit',
            'ilastik',
            'NanoPack',
            'OpenMM',
            'Gubbins',
            'Minimap2',
            'MetaPhlAn',
            'NextStrain',
            'TimeTree']

def statistcalTests():
    return [
        "Pearson correlation test, crating function named 'pearsonr'",
        "Spearman correlation test, creating function named 'spearmanr'",
        "Kendall’s Tau correlation test, creating function named 'kendalltau'",
        "Partial correlation test, creating function named 'partial_corr'",
        "Distance correlation test, creating function named 'dcor'",
        "Mantel test, creating function named 'mantel'",
        "Point-Biserial correlation test, creating function named 'pointbiserialr'",
        "Phi coefficient test, creating function named 'phicoeff'",
        "Chi-square test of independence, creating function named 'chi2indep'",
        "Chi-square goodness-of-fit test, creating function named 'chi2gof'",
        "Fisher’s exact test, creating function named 'fisher_exact'",
        "McNemar’s test, creating function named 'mcnemar'",
        "Cochran’s Q test, creating function named 'cochran_q'",
        "Barnard’s exact test, creating function named 'barnard_exact'",
        "Boschloo’s test, creating function named 'boschloo'",
        "Linear regression coefficient test, creating function named 'linreg'",
        "F-test for overall regression, creating function named 'f_test'",
        "Breusch-Pagan test, creating function named 'breusch_pagan'",
        "White test, creating function named 'white_test'",
        "Ramsey RESET test, creating function named 'reset_test'",
        "Durbin-Watson test, creating function named 'durbin_watson'",
        "Harvey-Collier test, creating function named 'harvey_collier'",
        "Variance Inflation Factor (VIF) test, creating function named 'vif'",
        "Shapiro-Wilk test, creating function named 'shapiro'",
        "Anderson-Darling test, creating function named 'anderson'",
        "Lilliefors test, creating function named 'lilliefors'",
        "Jarque-Bera test, creating function named 'jarque_bera'",
        "D’Agostino and Pearson’s test, creating function named 'dagostino'",
        "Levene’s test, creating function named 'levene'",
        "Bartlett’s test, creating function named 'bartlett'",
        "Brown-Forsythe test, creating function named 'brown_forsythe'",
        "Fligner-Killeen test, creating function named 'fligner_killeen'",
        "Hotelling’s T-squared test, creating function named 'hotelling_t'",
        "Permutation test, creating function named 'permutation_test'",
        "Permutation test for correlation, creating function named 'permutation_corr'"
    ]

def MCMCExamples():
    return [
        "Inferring gene regulatory networks from RNA-seq time series",
        "Detecting differential gene expression with low-count RNA-seq data",
        "Haplotype inference from genotypic data",
        "Estimating phylogenetic trees and divergence times",
        "Identifying genomic regions under selection in evolution experiments",
        "Calling copy number variations in cancer genomes",
        "Deconvolving bulk RNA-seq into cell-type proportions",
        "Inferring transcription factor binding kinetics from ChIP-seq",
        "Bayesian clustering of single-cell RNA-seq data",
        "Inferring latent variables in scRNA-seq using Bayesian autoencoders",
        "Estimating neural tuning curves with Bayesian regression",
        "Inferring connectivity in neural circuits from spike train data",
        "Bayesian decoding of neural activity for motor intent",
        "Detecting change points in neural firing patterns",
        "Modeling EEG/MEG sources using hierarchical Bayesian models",
        "Inferring rate constants from single-molecule FRET trajectories",
        "Parameter inference for nonlinear ODE models of gene circuits",
        "Bayesian inference of allosteric regulation mechanisms",
        "Estimating parameters in metabolic network models with flux data",
        "Using MCMC to identify bistability in gene regulatory networks",
        "Bayesian SIR model fitting to epidemic outbreak data",
        "Estimating time-varying reproduction number using MCMC",
        "Inferring under-reporting rates in COVID-19 or other outbreaks",
        "Bayesian meta-analysis of treatment efficacy across clinical trials",
        "Model-based Bayesian adaptive trial design",
        "Quantitative PET imaging with Bayesian hierarchical modeling",
        "Bayesian deconvolution of fluorescent microscopy data",
        "Inference of tumor growth parameters from longitudinal imaging",
        "Probabilistic segmentation of organs in medical images",
        "Parameter estimation in pharmacokinetic/pharmacodynamic (PK/PD) models",
        "Inferring reaction rate constants from reactor output using MCMC",
        "Bayesian design of experiments for catalytic systems",
        "Estimating parameters in adsorption isotherm models",
        "Inferring diffusion coefficients in porous media",
        "Bayesian calibration of process simulators in chemical manufacturing",
        "Inferring Monod parameters from batch fermentation data",
        "Bayesian optimization of microbial strain design",
        "Modeling yield variability in bioreactors using hierarchical models",
        "Parameter inference for biomass growth under uncertainty",
        "Bayesian control strategies for continuous biomanufacturing"
    ]
