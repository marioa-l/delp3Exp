import argparse
from models_generation.buildKBs import *
import glob
from utils.utils import *


class Datasets:
    
    # Some default configurations for models
    #
    # Analytical Model
    am_settings = {
            'simple': 'S',  # Dataset path of simples delp programs
            'medium' : 'M', # Dataset path of mediums delp programs
            'complex': 'C' # Dataset path of complex delp programs
            }
    # Annotation Function
    af_settings = {
            'simple':{
                'fa_ann': 0,    # Only kb base is annotated
                'var_use': 1 
                },
            'medium':{
                'fa_ann': 50,
                'var_use': 2
                },
            'complex':{
                'fa_ann': 100,
                'var_use': 3
                }
            }
    # Environmental Model
    em_settings = {
            'simple': {
                'var': 12,
                'var_use_annot': 12,
                'arcs': 12,  # To use Tup-Ind
                'alpha': 0.9,
                'tau': 1
                },
            'medium':{
                'var': 18,
                'var_use_annot': 18,
                'arcs': 18,
                'alpha': 0.9,
                'tau': 1    # Similar to tree structure
                },
            'complex':{
                'var': 24,
                'var_use_annot': 24,
                'arcs': 30,  
                'alpha': 0.6,
                'tau': 2
                } 
            }
    

    def __init__(self) -> None:
        pass    

    def build_models(self, am, af_set, em_set, output_path):
        """ To create setting for the experiment
        Args:
            -am: The dataset of delp programs
            -af_setting: 'simple', 'medium' or 'complex'
            -em_setting: 'simple', 'medium' or 'complex'
            -output_path: The path for save the delp3e models
        """
        creator = CreateDeLP3E(am, self.af_settings[af_set]['fa_ann'], 
                                    self.af_settings[af_set]['var_use'],
                                    self.em_settings[em_set]['var'],
                                    self.em_settings[em_set]['var_use_annot'],
                                    self.em_settings[em_set]['arcs'],
                                    self.em_settings[em_set]['alpha'],
                                    self.em_settings[em_set]['tau'],
                                    output_path)
        creator.create()


def get_all_programs(path: str) -> list:
    # Make the limit of files to return an option (input argument)
    return sorted(glob.glob(path + '*delp.json'), key=natural_key)[:300]


parser = argparse.ArgumentParser(description='Script to generate annotations \
                                                randomly for a del3e program')
parser.add_argument('-am',
                    action='store',
                    help="The path of the dataset with delp programs (json)",
                    dest="am",
                    type=get_all_programs,
                    required=True)
parser.add_argument('-af',
                    help='Annotation Function setting',
                    dest="af_set",
                    type=str,
                    required=True)
parser.add_argument('-em',
                    help='Environmental Model setting',
                    dest="em_set",
                    type=str,
                    required=True)
parser.add_argument('-out',
                    help='Path for the output file',
                    dest="output_path",
                    required=True)

arguments = parser.parse_args()

dataset = Datasets()
dataset.build_models(arguments.am, arguments.af_set, arguments.em_set, 
                                                        arguments.output_path)
