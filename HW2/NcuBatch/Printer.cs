using System;
using System.Collections.Generic;
using System.Text;

namespace NcuBatch
{
    class Printer
    {
        private IEnumerable<string> kernels = new List<string>() { "matAdd", "matAddRow", "matAddCol", "dgemv" };
        private IEnumerable<Tuple<int, int>> sizes = new List<Tuple<int, int>>
        {
            new Tuple<int, int>(4, 32),
            new Tuple<int, int>(128, 32),
            new Tuple<int, int>(1024, 32),
        };

        private string ncuTemplate = @"call ncu -k {0} --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_bytes.sum,lts__t_bytes.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__inst_executed_pipe_alu --log-file {1} --set default CudaRun.exe ";
        private string runTemplate = "call CudaRun.exe {0} ";
        private string nvprofTemplate = @"nvprof CudaRun.exe {0} ";

        public int ipow2(int e)
        {
            return 1 << e;
        }
        
        public void Print()
        {
            foreach (var kernel in kernels)
            {
                foreach (var size in sizes)
                {
                    for (int e = 8; e <= 26; e += 2)
                    {
                        var param = $"-k {kernel} -b {size.Item1} -t {size.Item2} -r {ipow2(e/2)}";
                        var output = $" 2> {kernel}-{size.Item1}-{size.Item2}-{ipow2(e / 2)}.txt";
                        Console.WriteLine(string.Format(nvprofTemplate, kernel) + param + output + System.Environment.NewLine);
                    }
                }
            }
        }
    }
}
