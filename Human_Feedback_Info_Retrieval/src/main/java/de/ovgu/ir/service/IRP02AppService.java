package de.ovgu.ir.service;

import java.io.IOException;
import java.util.List;

import org.apache.lucene.queryparser.classic.ParseException;

import de.ovgu.ir.model.QueryData;
import de.ovgu.ir.model.RerankData;

public interface IRP02AppService {
	List<QueryData> getSearchData(String query)throws ParseException, IOException;

	List<QueryData> getreRerankData(List<RerankData> obj) throws IOException, ParseException;
	
	public void logout();

}
