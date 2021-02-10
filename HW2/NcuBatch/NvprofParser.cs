using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace NcuBatch
{
    class NvprofParser
    {
        private Regex kernelRuntimeRegex = new Regex(".*(GPU activities:)? +(?<percent>.+%) +(?<time>\\d.+?[um]?s).+ (?<kernel>[a-zA-Z]+)$");
        private Regex memcpyHtoDRegex = new Regex(".*(GPU activities:)? +(?<percent>\\d.+%) +(?<time>\\d.+?[um]?s).+\\[CUDA memcpy HtoD\\]");
        private Regex memcpyDtoHRegex = new Regex(".*(GPU activities:)? +(?<percent>\\d.+%) +(?<time>\\d.+?[um]?s).+\\[CUDA memcpy DtoH\\]");
        private Regex fileNameRegex = new Regex("(?<kernel>.+?)-(?<size>\\d+?)-(?<threads>\\d+?)-(?<blocks>\\d+?).txt");

        public void ParseNvprof()
        {
            var files = Directory.GetFiles("D:\\src\\gpu2021hpc\\HW2\\hw2-part3-2").ToList();
            var txtFiles = files.Where(x => x.EndsWith(".txt"));

            Console.WriteLine("kernelName,size,threads,blocks,kernelRuntimePercent,kernelRuntimeTime,memcpyHtoDPercent,memcpyHtoDTime,memcpyDtoHPercent,memcpyDtoHTime");

            int i = 0;
            foreach(var txtFile in txtFiles)
            {
                var fileName = Path.GetFileName(txtFile);
                Match fileNameMatch = fileNameRegex.Match(fileName);
                string kernelName = fileNameMatch.Groups["kernel"].Value;
                string size = fileNameMatch.Groups["size"].Value;
                string threads = fileNameMatch.Groups["threads"].Value;
                string blocks = fileNameMatch.Groups["blocks"].Value;
                string kernelRuntimePercent = string.Empty;
                string kernelRuntimeTime = string.Empty;
                string memcpyHtoDPercent = string.Empty;
                string memcpyHtoDTime = string.Empty;
                string memcpyDtoHPercent = string.Empty;
                string memcpyDtoHTime = string.Empty;

                var lines = File.ReadLines(txtFile);
                var parse = false;
                foreach(var line in lines)
                {
                    if(line.Contains("Type"))
                    {
                        parse = true;
                    }
                    else if(line.Contains("API calls"))
                    {
                        parse = false;
                    }

                    if(parse)
                    {
                        var kernelRuntimeMatch = kernelRuntimeRegex.Match(line);
                        var memcpyHtoDMatch = memcpyHtoDRegex.Match(line);
                        var memcpyDtoHMatch = memcpyDtoHRegex.Match(line);

                        if(kernelRuntimeMatch.Success)
                        {
                            kernelRuntimePercent = kernelRuntimeMatch.Groups["percent"].Value;
                            kernelRuntimeTime = kernelRuntimeMatch.Groups["time"].Value;
                        }
                        else if(memcpyHtoDMatch.Success)
                        {
                            memcpyHtoDPercent = memcpyHtoDMatch.Groups["percent"].Value;
                            memcpyHtoDTime = memcpyHtoDMatch.Groups["time"].Value;
                        }
                        else if(memcpyDtoHMatch.Success)
                        {
                            memcpyDtoHPercent = memcpyDtoHMatch.Groups["percent"].Value;
                            memcpyDtoHTime = memcpyDtoHMatch.Groups["time"].Value;
                        }
                    }

                }

                var output = new List<string> { kernelName, size, threads, blocks, kernelRuntimePercent, kernelRuntimeTime, memcpyHtoDPercent, memcpyHtoDTime, memcpyDtoHPercent, memcpyDtoHTime };

                if(output.Any(x => string.IsNullOrEmpty(x)))
                {

                }

                Console.WriteLine(string.Join(",", output));
                i++;
            }
        }
    }
}
