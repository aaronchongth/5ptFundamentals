# Print the full directory path of the root of this project
function print_project_root {
  readlink -f `dirname $(readlink -f "$0")`/..
}


