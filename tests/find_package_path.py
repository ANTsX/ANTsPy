import importlib.util

# This is used for CI scripts to find the location of the installed package

package_name = 'ants'
package_spec = importlib.util.find_spec(package_name)
print(package_spec.submodule_search_locations[0] if package_spec else '')