package de.ovgu.ir.model;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;


public class DocumentIndexer {
	private Directory _IndexDirectory;
	private File _FilesDirectory;
	private Analyzer _Analyzer;
	private CustomIndexWriter _IndexWriter;

	public DocumentIndexer(String indexDirectory, String corpusDirectory) throws IOException {
		this._IndexDirectory = FSDirectory.open(Paths.get(indexDirectory));
		this._FilesDirectory = new File(corpusDirectory);
		this._Analyzer = new EnglishAnalyzer();
		this._IndexWriter = new CustomIndexWriter(this._IndexDirectory,	new IndexWriterConfig(this._Analyzer));
		this.ParsingAndIndexing();

	}

	public void ParsingAndIndexing() throws IOException {
		System.out.println("Parsing of Documents and Creation of Index");
		this._IndexWriter.CreateIndexFromFilesDirectory(this._FilesDirectory);
		this._IndexWriter.commit();
		this._IndexWriter.close();
		System.out.println("Indexes Created");
	}
}
