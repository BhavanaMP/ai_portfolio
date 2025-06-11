package de.ovgu.ir.model;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;


public class CustomIndexWriter extends IndexWriter{
	
	public CustomIndexWriter(Directory index, IndexWriterConfig config) throws IOException {
		super(index, config);
	}

	public void CreateIndexFromFilesDirectory(File corpusDirectory) throws IOException {
		HashSet<File> files = Helpers.GetFilesfromDirectory(corpusDirectory);
		Parser txtParser = new TxtParser();
		Parser htmlParser = new HtmlParser();
		for (File file : files) {
			Document document = new Document();
			ArrayList<Field> fields = null;
			if (file.getName().endsWith(".txt")) {
				fields = txtParser.parse(file);
			} 
			else if (file.getName().endsWith(".html") || file.getName().endsWith(".htm")) {
				fields = htmlParser.parse(file);
			}
			else 
				continue;
			
			for (Field field : fields) 
				document.add(field);

			this.addDocument(document);
		}
	}
}
