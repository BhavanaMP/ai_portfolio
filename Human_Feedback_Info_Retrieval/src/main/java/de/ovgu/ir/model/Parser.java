package de.ovgu.ir.model;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import org.apache.lucene.document.Field;

public interface Parser {
	public ArrayList<Field> parse(File file) throws IOException;
}
