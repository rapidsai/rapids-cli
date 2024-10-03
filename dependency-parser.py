# List of dependencies with version requirements
from packaging.version import Version
import importlib.metadata


def is_version_satisfied(package_name, required_version, operator):

    op_map = {
        "==" : lambda x, y : x == y,
        "<=" : lambda x, y : x <= y,
        ">=": lambda x, y : x >= y,
        ">" : lambda x, y : x > y,
        "<": lambda x, y : x < y
    }

    try:
        installed_version = importlib.metadata.version(package_name)
      
        if required_version: 
                # Strip the wildcard and consider the prefix only
            if required_version.endswith('.*'):
                base_version = required_version[:-2]
                base_version_num = Version(base_version)
                
                # Compare with the other version
                return op_map[operator](Version(installed_version), base_version_num), installed_version
                
            # If both versions are standard, compare them directly
            result =  op_map[operator](Version(installed_version), Version(required_version))
            return result, installed_version 
        return True, installed_version
            
    except importlib.metadata.PackageNotFoundError:
       
        return False, None

def dependency_parser(dependencies): 
    for dependency in dependencies:
        splitted = dependency.split(",")
        if len(splitted) == 2: 
            first_req, second_req = splitted
            if '*' in first_req: 
                first_req.replace('*', '0')
            if '*' in second_req: 
                second_req.replace('*', '0')
            operator1 = '>=' if '>=' in first_req else '<=' if '<=' in first_req else '=='  if '==' in first_req else '<' if '<' in first_req else '>' 
            
            name, version1 = first_req.split(operator1)

            
            operator2 = '>=' if '>=' in second_req else '<=' if '<=' in second_req else '==' if '==' in second_req else '<' if '<' in second_req else '>'
           
            version2 = second_req.split(operator2)[1]
        
            is_satisfied1, installed_version = is_version_satisfied(name, version1, operator1)
            is_satisfied2, installed_version= is_version_satisfied(name, version2, operator2)
            if is_satisfied1 and is_satisfied2:
                print(f'*{name}* dependency satisfied')
            elif not installed_version:
                print(f"*{name}* is not installed.")
            else:
                print(f'*{name}* dependency NOT satisfied. The current {name} version is {installed_version}. Please upgrade/downgrade to {dependency}')
        else:
            req = splitted[0]
            name = req
            if '*' in req: 
                req.replace('*', '0')
            operator = '>=' if '>=' in req else '<=' if '<=' in req else '=='  if '==' in req else '<' if '<' in req else '>' if '>' in req else None
            if not operator:
                is_satisfied, installed_version = is_version_satisfied(req, None, operator)
                if is_satisfied:
                    print(f'*{name}* dependency satisfied')
                else:
                    print(f"*{name}* is not installed.")
            else:
                name, version = req.split(operator)
                print(name, version)
                is_satisfied, installed_version = is_version_satisfied(name, version, operator)
                if is_satisfied:
                    print(f'*{name}* dependency satisfied')
                elif not installed_version:
                    print(f"*{name}* is not installed.")
                else:
                    print(f'*{name}* dependency NOT satisfied. The current {name} version is {installed_version}. Please upgrade/downgrade to {dependency}')
        
