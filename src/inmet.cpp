extern "C" int inmet(int argc, char **argv);

int main(int argc, char **argv)
{                   
    return inmet(argc - 1, argv + 1);
}
