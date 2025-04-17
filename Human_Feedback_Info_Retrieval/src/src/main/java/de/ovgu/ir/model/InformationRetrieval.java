package de.ovgu.ir.model;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;

import flexjson.JSONDeserializer;
import flexjson.JSONSerializer;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;

public class InformationRetrieval {
	private Directory _IndexDirectory;
	private File _FilesDirectory;
	private Analyzer _Analyzer;
	private IndexReader _IndexReader;
	private IndexSearcher _IndexSearcher;
	public InformationRetrieval(String indexDirectory) throws IOException {
		this._IndexDirectory = FSDirectory.open(Paths.get(indexDirectory));
		this._Analyzer = new EnglishAnalyzer();
		this._IndexReader = DirectoryReader.open(this._IndexDirectory);
		this._IndexSearcher = new IndexSearcher(this._IndexReader);
	}
	public String[][] RelevanceRetrieval(String queryStr) throws ParseException, IOException, ParseException {
		Query query = new MultiFieldQueryParser(FieldNames._QueryFieldNames, this._Analyzer).parse(queryStr);
		String fileName = new StringBuilder().append("\\Feedback\\").append(queryStr.toLowerCase()).append(".json").toString();
		TopDocs hits = this._IndexSearcher.search(query, Integer.MAX_VALUE);
		final Hashtable<String, Integer> my_rating = GetRatingFromFile(fileName);
		String[][] retrievalResults;
		String[][] retVal = null;

		float highestScore = Integer.MAX_VALUE;
		float lowestScore = 0;

		if (hits != null && hits.scoreDocs != null && hits.scoreDocs.length > 0) {
			retrievalResults = new String[hits.scoreDocs.length][7];
			highestScore = hits.scoreDocs[0].score;
			lowestScore = hits.scoreDocs[hits.scoreDocs.length - 1].score;

			for (int i = 0; i < hits.scoreDocs.length; i++) {
				Document document = _IndexReader.document(hits.scoreDocs[i].doc);

				String fullPath = document.get(FieldNames._FilePath);
				int lastIndex = fullPath.lastIndexOf('\\');
				List<Path> filesPaths = ListImageSourceFiles(fullPath.substring(0, lastIndex));

				retrievalResults[i][0] = Integer.toString(i + 1);
				if (filesPaths != null && filesPaths.size() > 0) {
					retrievalResults[i][1] = filesPaths.get(0).normalize().toString();
				} else {
					retrievalResults[i][1] = "No Image";
				}
				retrievalResults[i][2] = hits.totalHits.toString();
				retrievalResults[i][3] = Float.toString(hits.scoreDocs[i].score);//Score
				if (document.get(FieldNames._Title) != null)
					retrievalResults[i][4] = document.get(FieldNames._Title);
				if (document.get(FieldNames._FilePath) != null)
					retrievalResults[i][5] = document.get(FieldNames._FilePath);
				;//Score
				if (document.get(FieldNames._Title) != null) {
					if (my_rating.containsKey(retrievalResults[i][4])) {
						long rating = my_rating.get(retrievalResults[i][4]);
						if (rating > 0 && rating < 3) {
							retrievalResults[i][3] = Float.toString(Float.parseFloat(retrievalResults[i][3]) - lowestScore - rating);
						} else if (rating > 3) {
							retrievalResults[i][3] = Float.toString(highestScore + rating);
						}
					}
				}

				if (document.get(FieldNames._Title) != null) {
					if (my_rating.containsKey(retrievalResults[i][4])) {
						long rating = my_rating.get(retrievalResults[i][4]);
						retrievalResults[i][6] = Long.toString(rating);
					}
					else
					{
						retrievalResults[i][6] = Long.toString(0);
					}
				}
			}
			java.util.Arrays.sort(retrievalResults, new java.util.Comparator<String[]>() {
				public int compare(String[] a, String[] b) {
					final String str1 = b[3];
					final String str2 = a[3];
					return str1.compareTo(str2);
				}
			});
			if (retrievalResults.length > 10)
			{
				retVal= new String[10][7];
			}
			else
			{
				retVal= new String[retrievalResults.length][7];
			}
			for (int i = 0; i < 10 && i< retrievalResults.length; i++) {
				retVal[i] = retrievalResults[i];
			}
		}
		return retVal;
	}

	public void ResetSystem(){
		try{
			FileUtils.cleanDirectory(new java.io.File("\\Feedback"));
		}
		catch (Exception e){

		}
	}

	List<Path> ListImageSourceFiles(String location) throws IOException {
		Path dir = Paths.get(location);
		List<Path> result = new ArrayList<>();
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.{png,jpeg,jpg,JPG,gif,JPEG,GIF}")) {
			for (Path entry: stream) {
				result.add(entry);
			}
		} catch (DirectoryIteratorException ex) {
			// I/O error encounted during the iteration, the cause is an IOException
			throw ex.getCause();
		}
		return result;
	}

	public Hashtable<String, Integer> GetRatingFromFile(String fileName) {
		Hashtable<String, Integer> my_dict = null;
		try {
			System.out.println(fileName);
			String feedback = Files.readString(Path.of(fileName));
			//System.out.println(Path.of(fileName));
			//String feedback = Files.readString(Path.of(fileName));
			//String feedback = new String(Files.readAllBytes(Path.of(fileName)));
			JSONDeserializer<HashMap<String,Integer>> deSerializer = new JSONDeserializer<HashMap<String, java.lang.Integer>>();
			HashMap<String,Integer> temp = deSerializer.deserialize(feedback);
			if(temp!=null && temp.keySet()!= null && temp.keySet().size()>0)
			{
				my_dict = new Hashtable<String, Integer>();
				for (String key:temp.keySet())
				{
					my_dict.put(key,temp.get(key));
				}
			}
		}
		catch (Exception ex)
		{
			my_dict = new Hashtable<String, Integer>();
		}
		return my_dict;
	}

	public void SaveFeedbackRating(List<RerankData> data) throws IOException {
		String queryStr = null;
		Hashtable<String,Integer> hash =null;

		if(data!=null && data.size() >0)
		{
			hash = new Hashtable<String,Integer>();
			queryStr = data.get(0).query;
			for(RerankData item:data)
			{
				hash.put(item.title,item.ratings);
			}

		}

		String fileName = new StringBuilder().append("\\Feedback\\").append(queryStr.toLowerCase()).append(".json").toString();
		Path pathToFile = Paths.get(fileName);
		 Hashtable<String,Integer> my_dict = GetRatingFromFile(fileName);
		if(hash!=null){
			Hashtable<String, Integer> finalHash = hash;
			hash.forEach((k, v) -> {
						if(my_dict.containsKey(k)){
							my_dict.replace(k, finalHash.get(k));
						}
						else
						{
							my_dict.put(k,v);
						}
					}
			);
			JSONSerializer serializer = new JSONSerializer();
			String json = serializer.serialize(my_dict);
			//File file = new File(fileName);
			try
			{

				Files.createDirectories(pathToFile.getParent());
				File file = new File(fileName);
				file.createNewFile();
				//Files.createFile(pathToFile);
			}
			catch (IOException e)
			{
				e.printStackTrace();    //prints exception if any
			}
			FileWriter writer = new FileWriter(fileName);
			writer.write(json);
			writer.close();
		}
	}
}
