package de.ovgu.ir.service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.queryparser.classic.ParseException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import de.ovgu.ir.configuration.IRP02AppConfig;
import de.ovgu.ir.model.InformationRetrieval;
import de.ovgu.ir.model.QueryData;
import de.ovgu.ir.model.RerankData;

@Service
public class IR02AppServiceImp implements IRP02AppService {
	private IRP02AppConfig config;
	private String IndexDirectory ;
	private InformationRetrieval ir;
	private List<QueryData> resultlist;
	@Autowired
	public IR02AppServiceImp(IRP02AppConfig config) throws IOException {
		this.config = config;
		this.IndexDirectory = "./Index";
		ir = new InformationRetrieval(IndexDirectory);
	}
	@Override
	public List<QueryData> getSearchData(String query) throws ParseException, IOException { 
		resultlist = new ArrayList<>();
		String[][] results = this.ir.RelevanceRetrieval(query);
		if (results != null){
			for (int i = 0; i < results.length; i++) {
				resultlist.add(new QueryData(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5], results[i][6]));
			}
		}
	
		
		return resultlist;
	}
	@Override
	public List<QueryData> getreRerankData(List<RerankData> list) throws IOException, ParseException {
		ir.SaveFeedbackRating(list);
		resultlist = new ArrayList<>();
		String query = list.get(0).query;
		System.out.println(query);
		String[][] results = this.ir.RelevanceRetrieval(query);
		if (results != null){
			for (int i = 0; i < results.length; i++) {
				resultlist.add(new QueryData(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5], results[i][6]));
				
			}
		}
		for(QueryData q:resultlist){
			System.out.println(q.getTitle());
			System.out.println(q.getRatings());
		}
		return resultlist;
	}
	@Override
	public void logout() {
		this.ir.ResetSystem();
		return;
	}
}
