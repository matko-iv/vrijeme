// ML Forecasting Pipeline by Matija Ivanovic(TM) - launcher.
// Starts the bundled Python GUI that sits next to this exe. Nothing is
// extracted to or installed on the host PC; every path is relative to the
// folder this exe lives in, so the drive letter does not matter.
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("ML Forecasting Pipeline by Matija Ivanovic™")]
[assembly: System.Reflection.AssemblyProduct("ML Forecasting Pipeline")]
[assembly: System.Reflection.AssemblyCompany("Matija Ivanovic")]
[assembly: System.Reflection.AssemblyCopyright("Matija Ivanovic™")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]

static class Launcher
{
    [STAThread]
    static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string pyw = Path.Combine(root, "runtime", "python", "pythonw.exe");
        string gui = Path.Combine(root, "app", "gui", "pipeline_gui.py");
        if (!File.Exists(pyw) || !File.Exists(gui))
        {
            MessageBox.Show(
                "The pipeline files are missing next to this program.\n\nExpected:\n" + pyw + "\n" + gui +
                "\n\nCopy the whole MLForecastPipeline folder, not just the exe.",
                "ML Forecasting Pipeline", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        string extra = "";
        foreach (string a in args) extra += " \"" + a.Replace("\"", "") + "\"";
        var psi = new ProcessStartInfo(pyw, "\"" + gui + "\"" + extra);
        psi.WorkingDirectory = root;
        psi.UseShellExecute = false;
        psi.EnvironmentVariables.Remove("PYTHONPATH");
        psi.EnvironmentVariables.Remove("PYTHONHOME");
        psi.EnvironmentVariables.Remove("VIRTUAL_ENV");
        psi.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        try
        {
            Process.Start(psi);
        }
        catch (Exception e)
        {
            MessageBox.Show("Could not start the pipeline: " + e.Message, "ML Forecasting Pipeline",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 2;
        }
        return 0;
    }
}
