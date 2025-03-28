import importlib
import pkgutil
import sionna

def explore_package(package, indent=0):
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        print("  " * indent + f"{'[PKG]' if ispkg else '[MOD]'} {modname}")
        if ispkg:
            try:
                subpkg = importlib.import_module(f"{package.__name__}.{modname}")
                explore_package(subpkg, indent + 1)
            except Exception as e:
                print("  " * (indent + 1) + f"⚠️ {modname} could not be imported: {e}")

if __name__ == "__main__":
    print("🔍 Exploring Sionna modules:")
    explore_package(sionna)
