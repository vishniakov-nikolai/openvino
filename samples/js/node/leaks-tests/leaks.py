import openvino.runtime as ov

def test():
  print(dir(ov))

def create_and_release_core():
  core = ov.Core()
  core.get_versions("CPU")

def create_and_release_model():
  core = ov.Core()

  for i in range(0, 10000):
    model = core.read_model('../../assets/models/classification.xml')

def main():
  for i in range(0, 300_000):
    create_and_release_core()
  # create_and_release_model()


main()
# test()
