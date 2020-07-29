package = "tvnorm-nn"
version = "scm-1"

source = {
    url = "https://github.com/pengsun/tvnorm-nn",
}

description = {
    summary = "Total Variation Norm as Torch 7 nn module",
    detailed = [[
        Sobel -> Abs -> AveragePooling
    ]],
    homepage = "https://",
    license = "MIT"
}

dependencies = {
    "torch >= 7.0",
}

build = {
    type = "command",
    build_command = [[
        cmake -E make_directory build;
        cd build;
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)";
        $(MAKE)
    ]],
    install_command = "cd build && $(MAKE) install"
}