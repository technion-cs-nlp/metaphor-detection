import yaml
import sys

# f = open("config/config.yaml")
# y = yaml.safe_load(f)
# for i in range(1, len(sys.argv)):
#     section = "model_args"  # default section
#     var, val = sys.argv[i].split("=")
#     var_splitted = var.split(':')
#     if len(var_splitted) == 2:
#         section, var = var_splitted
#     if var.find("_int") != -1:
#         var = var.split('_int')[0]
#         y[section][var] = int(val)
#     elif var.find("_float") != -1:
#         var = var.split('_float')[0]
#         y[section][var] = float(val)
#     else:
#         y[section][var] = val
# f = open("config/config.yaml", "w")
# yaml.dump(y, f)
# f.close()
#
#
# def set_specific_config(yaml_file, section, var, val):
#     f = open(yaml_file)
#     y = yaml.safe_load(f)
#     y[section][var] = val
#     f = open(yaml_file, "w")
#     yaml.dump(y, f)
#     f.close()

