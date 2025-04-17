package de.ovgu.ir.model;
import java.io.File;
import java.io.IOException;
import java.text.Format;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;

public class Helpers {
    public static String ConvertLongToString(long time){
        Date date = new Date(time);
        Format format = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        return format.format(date);
    }

    public static HashSet<File> GetFilesfromDirectory(File dir) throws IOException {
        HashSet<File> files = new HashSet<File>();

        File[] listFiles = dir.listFiles();

        for (File file :  listFiles) {
            if(file.isDirectory())
                files.addAll(GetFilesfromDirectory(file));
            else
                files.add(file);

        }
        return files;
    }
}
