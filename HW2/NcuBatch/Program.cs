using System;

namespace NcuBatch
{
    class Program
    {
        static void Main(string[] args)
        {
            //var printer = new Printer();
            //printer.Print();

            var parser = new NvprofParser();

            parser.ParseNvprof();
        }
    }
}
