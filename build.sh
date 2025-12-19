if [ "$#" -ge 1 ]; then
    cmake --build build -j --target $1
else
    cmake --build build -j
fi
