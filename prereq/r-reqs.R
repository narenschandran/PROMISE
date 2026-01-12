pkgs <- c('getopt', 'R.utils', 'data.table')

inst <- rownames(installed.packages())

mis  <- setdiff(pkgs, inst)

if (length(mis) > 0) install.packages(mis)

