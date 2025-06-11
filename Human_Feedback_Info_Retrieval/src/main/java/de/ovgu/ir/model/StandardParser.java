package de.ovgu.ir.model;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;

public class StandardParser implements Parser {

	@Override
	public ArrayList<Field> parse(File file) throws IOException {
		ArrayList<Field> fields = new ArrayList<Field>();

		String filename = file.getName();
		fields.add(new StoredField(FieldNames._FileName, filename));
		
		String path = file.getCanonicalPath();
		fields.add(new StoredField(FieldNames._FilePath, path));
		
		String lastModified = Helpers.ConvertLongToString(file.lastModified());
		fields.add(new StoredField(FieldNames._LastModifiedDate, lastModified));
		
		return fields;
	}

}
