package de.ovgu.ir.model;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;

public class TxtParser extends StandardParser {
	@SuppressWarnings("deprecation")
	@Override
	public ArrayList<Field> parse(File file) throws IOException {
		ArrayList<Field> fields = new ArrayList<Field>();
		fields.addAll(super.parse(file));
		String title = file.getName().substring(0,file.getName().indexOf('.'));
		fields.add(new TextField(FieldNames._Title, title, Field.Store.YES));
		String text = FileUtils.readFileToString(file);
		fields.add(new TextField(FieldNames._Content, text, Field.Store.YES));

		return fields;

	}
}
