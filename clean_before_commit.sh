echo 'clean .pyc files...'
find . -name "*.pyc" -type f -delete;

echo 'clean .ide folders...'
find . -name ".idea" -type d -a -prune -exec rm -rf {} \;

echo 'clena ._* files...'
find . -name "*._*" -type f -delete;
