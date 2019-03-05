extern "C" int inmet_aux(int argc, char **argv);

int main(int argc, char **argv)
{
    return inmet_aux(argc - 1, argv + 1);
}
