from setup import find_packages, setup


Hypen='-e .'
def get_requirements(file_path):
    with open(file_path, 'r') as file:
        # Read the file and remove any hyphenated entries
        requirements = [line.strip() for line in file if line.strip() and line.strip() != Hypen]
    return requirements

setup(
    name='MLProject1',
    version='0.1.0',
    author='Kodavati vivek',
    author_email='vivekchowdary457@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)