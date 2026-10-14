# Private stdin-lifetime handle holder. No write/delete sharing for file OR ancestors.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class SourceLeaseHandle {
 [StructLayout(LayoutKind.Sequential)] struct Info {
  public uint attributes; public uint creationLow; public uint creationHigh;
  public uint accessLow; public uint accessHigh; public uint writeLow; public uint writeHigh;
  public uint volume; public uint sizeHigh; public uint sizeLow; public uint links;
  public uint indexHigh; public uint indexLow;
 }
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)]
 static extern SafeFileHandle CreateFile(string name,uint access,uint share,IntPtr security,uint creation,uint flags,IntPtr template);
 [DllImport("kernel32.dll",SetLastError=true)]
 static extern bool GetFileInformationByHandle(SafeFileHandle handle,out Info info);
 public static SafeFileHandle Open(string name,bool directory) {
  var h=CreateFile(name,directory?0u:0x80000000u,1,IntPtr.Zero,3,directory?0x02200000u:0x00200000u,IntPtr.Zero);
  if(h.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());
  Info info;
  if(!GetFileInformationByHandle(h,out info)) { h.Dispose();throw new Win32Exception(Marshal.GetLastWin32Error()); }
  if((info.attributes&0x400)!=0) { h.Dispose();throw new InvalidOperationException("reparse point needs strict authentication"); }
  return h;
 }
}
'@
$files = @{}
$directories = @{}
function Drop-Directories($names) {
    foreach ($name in $names) {
        $entry = $directories[$name]
        $entry.refs--
        if ($entry.refs -eq 0) { $entry.handle.Dispose(); $directories.Remove($name) }
    }
}
function Close-Lease($id) {
    if ($files.ContainsKey($id)) {
        $entry=$files[$id];$entry.stream.Dispose()
        Drop-Directories $entry.directories
        $files.Remove($id)
    }
}
try {
    while ($null -ne ($line = [Console]::ReadLine())) {
        try {
            $request = ConvertFrom-Json -InputObject $line
            switch ($request.op) {
                'open' {
                    $path=[IO.Path]::GetFullPath([string]$request.path)
                    $parent=[IO.DirectoryInfo]::new([IO.Path]::GetDirectoryName($path))
                    $parents=@()
                    while ($null -ne $parent) { $parents=@($parent.FullName)+$parents; $parent=$parent.Parent }
                    $held=@();$stream=$null
                    try {
                        foreach ($name in $parents) {
                            if (!$directories.ContainsKey($name)) {
                                $directories[$name]=@{handle=[SourceLeaseHandle]::Open($name,$true);refs=0}
                            }
                            $directories[$name].refs++;$held+=$name
                        }
                        $handle=[SourceLeaseHandle]::Open($path,$false)
                        try { $stream=[IO.FileStream]::new($handle,[IO.FileAccess]::Read) }
                        catch { $handle.Dispose();throw }
                        $hash=[Security.Cryptography.SHA256]::Create()
                        try { $digest=[BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
                        finally { $hash.Dispose() }
                        $files[[string]$request.id]=@{stream=$stream;directories=$held}
                        $reply=@{ok=$true;digest=$digest;path_protected=$true}
                    } catch {
                        if ($null -ne $stream) { $stream.Dispose() }
                        Drop-Directories $held
                        throw
                    }
                }
                'close' { Close-Lease ([string]$request.id);$reply=@{ok=$true} }
                'ping' { $reply=@{ok=$true;count=$files.Count} }
                default { throw 'unknown private lease operation' }
            }
        } catch { $reply=@{ok=$false;error=$_.Exception.Message} }
        [Console]::WriteLine((ConvertTo-Json -InputObject $reply -Compress))
    }
} finally {
    foreach ($entry in $files.Values) { $entry.stream.Dispose() }
    foreach ($entry in $directories.Values) { $entry.handle.Dispose() }
}
